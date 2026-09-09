import os

import torch.nn as nn
import torch
import math
import numpy as np
import tqdm

from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

from RUnet import RUEncoder, RUDecoder

from Basicblocks import res2dblocks, res3dblocks
from Res_3D import Res3D_net
from Basic import video_frames

from loss_utils import gradient_loss, gradient_metric, SquaredFrobeniusLoss, EWMA

from config import seed
RANDOM_SEED = seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class MM_Model(nn.Module):
    def __init__(self,
                 AE_channel_in=12, AE_channel_out=3, AE_struct=[1, 1, 1], AE_layer_nums=3,
                 temporal_channel_in=1, temporal_struct=[1, 1, 1], temporal_layer_nums=3,
                 img_channels=3,
                 frame_nums=5, compact = False):
        super(MM_Model, self).__init__()

        self.compact = compact
        self.ewma = EWMA(betha=0.9)
        self.norm = F.normalize

        self.img_channel = img_channels
        self.frame_nums = frame_nums

        self.neck_planes = 32
        self.inter_planes = self.neck_planes * (2 ** AE_layer_nums)

        # define sub models
        self.spatial_encoder = RUEncoder(res2dblocks, input_channels=AE_channel_in, layers=AE_struct,
                                         layer_num=AE_layer_nums, neck_planes=self.neck_planes, last_layer_softmax=True)
        self.temporal_func = Res3D_net(res3dblocks, input_channels=1, layers=temporal_struct, layer_num=3,
                                       neck_planes=self.neck_planes, last_layer_softmax=True)
        self.spatial_decoder = RUDecoder(res2dblocks, output_channels=AE_channel_out, layers=AE_struct[::-1],
                                         layer_num=AE_layer_nums, neck_planes=self.neck_planes, )


        # model parameters
        self.model_par = list(self.spatial_encoder.parameters()) \
                         + list(self.temporal_func.parameters()) \
                         + list(self.spatial_decoder.parameters())

        # cmp parameters list(self.temporal_func.parameters()) +
        # self.cmp_par =  list(self.temporal_func.flow[-1].parameters()) # list(self.spatial_encoder.layer_list[-1].last_layer.parameters())  + 
        # self.cmp_par = list(self.spatial_encoder.layer_list.parameters()) + list(self.temporal_func.parameters()) 
        # self.cmp_par = list(self.spatial_encoder.layer_list[-1].last_layer.parameters())
        # list(self.spatial_encoder.layer_list[-1].last_layer.parameters())
        self.cmp_par = list(self.spatial_encoder.layer_list.parameters())

        # model criterion
        self.l2_criterion = nn.MSELoss()
        self.l3_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.fro_loss_DW = SquaredFrobeniusLoss()
        self.M_DW = None
        self.M_SW = None
        self.temporal_samples = []

    def forward(self, x, stage=['G'], iter_idx=0):
        loss_cmp = []
        if not stage:
            stage = ['G']

        # get frames
        static_in, motion_in, static_target, motion_target = video_frames(x, self.img_channel, self.frame_nums)
        encoder = self.spatial_encoder(motion_in)
        temp_in = encoder[0].unsqueeze(dim=1)
        temporal = self.temporal_func(temp_in)
        # temporal = temporal.permute(0, 2, 1, 3, 4).contiguous()
        # temporal = temporal.squeeze(dim=1)


        if 'G' in stage:

            encoder[0] = torch.cat( [ encoder[0], temporal ] ,1 )
            # decoder_in = encoder
            # decoder_in[0] = temporal

            decoder = self.spatial_decoder(encoder) # - static_in

            pred_target = motion_target  #  + static_in

            loss_predict = self.l2_criterion(decoder, pred_target)
            grad_predict = gradient_loss(decoder, pred_target)

            return loss_predict, grad_predict

        if 'C' in stage:
            # self.sample_mean = torch.mean(temporal, dim=1, keepdim=False)  # , keepdim=True)
            # self.M =  self.ewma(torch.mean(self.sample_mean, dim=0, keepdim=False)) # update mean

            # update mean

            # self.M = ((self.M * iter_idx) + self.sample_mean) / (iter_idx + 1)
            # loss_cmp = self.l2_criterion(torch.mean(temporal,dim=1, keepdim=True), self.ewma.bias_corrected())
            # temporal = self.norm(temporal,p=2, dim=1)
            samples = temporal.detach().clone()
            # samples = temp_in.detach().clone()
            ##### depth_wise mean #####
            sample_mean_DW = torch.mean(samples, dim=1, keepdim=False) #, keepdim=True)
            self.M_DW = self.ewma(torch.mean(sample_mean_DW, dim=0, keepdim=True))
            loss_cmp_DW = self.fro_loss_DW(temporal, self.M_DW.unsqueeze(0))

            # loss_cmp_DW = self.fro_loss_DW(temporal, self.M_DW)

            ##### spatial_wise mean #####
            # sample_mean_SW = torch.mean(samples, dim=(2, 3), keepdim=True)
            # self.M_SW = self.ewma(torch.mean(sample_mean_SW, dim=0, keepdim=True))

            # loss_cmp_SW = self.fro_loss_SW(temporal, self.M_SW)
            
            return loss_cmp_DW

            # return loss_cmp_DW, loss_cmp_SW

            # loss_cmp = self.fro_loss(temporal, self.M) # self.fro_loss(input, Target)
            # temporal_dif = temporal - self.M
            # loss_cmp = torch.mean(torch.norm(temporal_dif, dim=[2, 3], p='fro')**2)
            # return loss_cmp


        if 'E' in stage:
            encoder = self.spatial_encoder(motion_in)

            temp_in = encoder[0].unsqueeze(dim=1)
            temporal = self.temporal_func(temp_in)

            # dist = torch.mean(torch.norm(temporal - self.M_DW, dim=[2,3],p='fro'))
            # temporal = temporal.permute(0, 2, 1, 3, 4).contiguous()
            # temporal = temporal.permute(0, 2, 1, 3).contiguous()

            # self.temporal_samples.append(temporal.detach().cpu().numpy())
            # if len(self.temporal_samples) > 0:
            #     Final = np.concatenate(self.temporal_samples, axis=0)
            #     np.save("temporal_samples.npy", Final)

            # temporal = temporal.squeeze(dim=1)
            encoder[0] = torch.cat( [ encoder[0], temporal ] ,1 )
            # decoder_in = encoder
            # decoder_in[0] = temporal

            decoder = self.spatial_decoder(encoder) # - static_in

            return decoder # , dist

# if __name__ == "__main__":
#
#     x = torch.randn(2, 15,256,256)
#     model_x = MM_Model()
#
#     print(model_x)
#     decoder, temporal, encoder, loss_predict, grad_predict = model_x(x)
