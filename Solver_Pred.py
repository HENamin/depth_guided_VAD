import os
import gc
import torch.nn as nn
import torch
import math
import numpy as np
from numpy import log10
import tqdm
import torch.optim as optim

from eval_utils import loss_map_re, calcu_result, reciprocal_metric

from Basic import video_frames
from Model import MM_Model

import config
from config import seed

from depth_map import depth_map


class Solver():
    def __init__(self, config, Model=MM_Model):
        self.log_dir = config.log_path
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_path = 'model.pth'
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_channel = config.img_channel
        self.frames_num = config.clips_length
        self.model = Model(
            AE_channel_in=self.img_channel * (self.frames_num - 1),
            AE_channel_out=self.img_channel,
            AE_struct=config.AE_struct,
            AE_layer_nums=config.AE_layer_nums,
            temporal_channel_in=config.temporal_channel_in,
            temporal_struct=config.temporal_struct,
            temporal_layer_nums=config.temporal_layer_nums,
            img_channels=self.img_channel,
            frame_nums=self.frames_num,
            compact=config.compact,
            # cmp_mode=config.cmp_mode,
        ).to(self.device)
        self.init_info()
        self.l2_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        self.optimizer = optim.Adam(self.model.model_par, lr=1e-5)
        self.optimizer_cmp = optim.Adam(self.model.cmp_par, lr=1e-5)
        self.lr_start = 1e-5
        self.lr_end = 1e-6
        num_epochs = 20
        gamma = (self.lr_end / self.lr_start) ** (1 / num_epochs)
        self.scheduler_cmp = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_cmp, gamma=gamma)

    def train_batch_cmp(self, batch_in, alpha=None, loss_appendix=0):
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        self.model.zero_grad()
        batch_in = batch_in.to(self.device)
        loss_cmp = self.model(batch_in, ['C'])
        loss_cmp.mean().backward(retain_graph=True)
        self.optimizer_cmp.step()
        self.optimizer_cmp.zero_grad()
        self.info['compact_loss'].append(loss_cmp.mean().item())
        return

    def train_batch_AE(self, batch_in, alpha=None, loss_appendix=0):
        self.model.train()
        self.model.zero_grad()
        batch_in = batch_in.to(self.device)
        loss_predict, grad_predict = self.model(batch_in, ['G'])
        loss = (loss_predict + 0.01 * grad_predict)
        loss.mean().backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        psnr_predict = 10 * log10(1 / loss_predict.mean().item())
        self.info['psnr_predict'].append(psnr_predict)
        self.info['total_loss'].append(loss.mean().item())
        return

    def training_info(self, detail_info):

        for info_keys in self.info.keys():
            if not self.info[info_keys] == []:
                detail_info += ' \t {} : {:.5f} '.format(info_keys, np.stack(self.info[info_keys]).mean())
        detail_info += '\n'
        self.init_info()
        print(detail_info)
        with open(os.path.join(self.log_dir, 'training_log.txt'), 'a+') as f:
            f.writelines(detail_info)
        return

    def eval_datasets(self, dataloader, labels_list, epoch=0):
        self.model.eval()
        eval_metric_dict = {}
        eval_metric_dict['inv_recon'] = []
        depth = None
        depth_normal = None

        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(dataloader.fetch_nums)):
                batch_in = dataloader.fetch()
                batch_in = batch_in.to(self.device)

                static_in, motion_in, static_target, motion_target = video_frames(batch_in, self.img_channel,
                                                                                  self.frames_num)
                self.model.zero_grad()
                if self.para_tag:
                    static_decoder, pred_decoder, loss_cluster_map = nn.parallel.data_parallel(self.model,
                                                                                               (batch_in, None, ['E']),
                                                                                               device_ids=self.device_ids)
                else:
                    batch_in = batch_in.to(self.device)
                    pred_decoder = self.model(motion_in, ['E'])

                pred_recon = pred_decoder
                pred_target = motion_target
                if True:
                    if config.rgb_tags:
                        Depth_in = motion_in[:, 9:12, :, :]  # for RGB datasets which they have ch = 3
                    else:
                        Depth_in = motion_in[:, 0:3, :, :]  # for Ped2 which is gray scale with ch = 1

                    depth = depth_map(Depth_in)

                    eps = 1e-8
                    min_val = depth.amin(dim=[2, 3], keepdim=True)
                    max_val = depth.amax(dim=[2, 3], keepdim=True)
                    depth_normal = (depth - min_val) / (max_val - min_val + eps)

                    depth_normal = torch.clamp(depth_normal, min=eps)
                    lss = (pred_recon - pred_target) * torch.log(depth_normal + eps)
                else:
                    lss = (pred_recon - pred_target)

                loss_pixelwise_re = loss_map_re(torch.mean((lss) ** 2, [1], keepdim=True))
                eval_metric_dict['inv_recon'].append(reciprocal_metric(loss_pixelwise_re))

        auc_list = []
        for eval_keys in eval_metric_dict.keys():
            eval_metric = np.concatenate(eval_metric_dict[eval_keys])
            auc, norm_result = calcu_result(eval_metric, labels_list, converse=False)

        eval_metric_dict['labels'] = labels_list
        detail_info = 'Epoches {} \t  auc {:.5f} \n '.format(epoch, auc)

        print(detail_info)
        if True:
            from scipy.io import savemat
            savemat('eval_metric.mat', {'data': eval_metric})
            savemat('norm_result.mat', {'data': norm_result})
            savemat('labels_list.mat', {'data': labels_list})

            # np.save('eval_metric',eval_metric )
            # np.save('norm_result',norm_result )
            # np.save('labels_list',labels_list )
        return

    def init_info(self):
        self.info = {}
        self.info['total_loss'] = []
        self.info['psnr_predict'] = []
        self.info['compact_loss'] = []
        return

    def load_model(self, eval_only=True):
        """Load model for evaluation or training"""
        load_path = os.path.join(self.log_dir, self.checkpoint_path)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            if not eval_only:
                # Load optimizer states if not evaluation only
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'optimizer_cmp' in checkpoint:
                    self.optimizer_cmp.load_state_dict(checkpoint['optimizer_cmp'])
                self.start_epoch = checkpoint.get('epoch', 0)
                print(f"Resumed from iteration {self.start_epoch}")
            else:
                print(f"Model loaded for evaluation from {load_path}")

            if 'M_dict' in checkpoint:
                self.model.M_SW = checkpoint['M_dict']
        else:
            print(f"No checkpoint found at {load_path}")

    def save_model(self, epoch):
        """Save checkpoint with all necessary states"""
        state = {
            'epoch': epoch + 1,  # Save next epoch to resume from
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_cmp': self.optimizer_cmp.state_dict(),
            'M_dict_SW': self.model.M_SW,
            'M_dict_DW': self.model.M_DW
        }

        # Save epoch-specific checkpoint
        save_path = os.path.join(self.log_dir, f'E{epoch}_{self.checkpoint_path}')
        torch.save(state, save_path)
        # Also save as latest checkpoint for resume
        latest_path = os.path.join(self.log_dir, self.checkpoint_path)
        torch.save(state, latest_path)
        print(f"Checkpoint saved to {save_path}")
        self.model.M_dict = {}  # Clear M_dict after saving
        return
