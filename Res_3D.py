import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Basicblocks import res3dblocks, conv3x3x3
from loss_utils import EWMA

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class Conv3DSimple(nn.Conv3d):
    expansion = 1
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3,3,3),
            # kernel_size=(63, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride

# class BasicBlock(nn.Module):
#
#     expansion = 1
#
#     def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
#         midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
#
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#             conv_builder(inplanes, planes, midplanes, stride),
#             nn.BatchNorm3d(planes),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             conv_builder(planes, planes, midplanes),
#             nn.BatchNorm3d(planes)
#         )
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


# class EWMA():
#     def __init__(self, betha):
#         super(EWMA,self).__init__()
#         self.betha = betha
#         self.ema = None  # do we need to initiaize with zero or the first value ?
#         self.step = 0 # counts the Updates
#
#     def update(self, value):
#         self.step += 1
#         if self.ema is None:
#             self.ema = value.clone().detach()  # Initialize with the first value
#         else:
#             self.ema = self.betha * self.ema + (1 - self.betha) * value
#
#     # def bias_corrected (self):
#         bias_correction  = 1 - (self.betha ** self.step)
#         if (bias_correction != 0):
#             return self.ema / bias_correction
#         else:
#             return self.ema


class Res3D_net(nn.Module):
    def __init__(self, block, input_channels = 1, neck_planes  = 32, layers=[1, 1, 1], layer_num= 3, last_layer_softmax = False, bn_tag=False):
        super(Res3D_net, self).__init__()

        self.neck_planes = neck_planes*2
        self.inplanes = self.neck_planes//2
        self.layers = layers
        self.last_layer_softmax = last_layer_softmax
        self.norm = F.normalize
        self.layer_num = layer_num
        # self.ewma = EWMA(0.9)
        # self.i = 0
        # self.m = None



        layer0 = []
        self.flow = nn.Sequential()
        layer0.append(nn.Conv3d(in_channels=input_channels, out_channels=self.inplanes, kernel_size=(3,3,3), padding=(1,1,1), stride=(2, 1, 1), bias=False))
        # layer0.append(nn.Conv3d(in_channels=input_channels, out_channels=self.inplanes, kernel_size=(127,3,3), padding=(63,1,1), stride=(2, 1, 1), bias=False))

        layer0.append(nn.ReLU(inplace=True))

        self.flow.add_module('layer0',nn.Sequential(*layer0))

        for layer_idx in range(self.layer_num):
            # self.flow.add_module('bn_{}'.format(layer_idx), nn.BatchNorm3d(self.inplanes))
            self.flow.add_module('motion_{}'.format(layer_idx), self._make_layer(block, self.neck_planes*(2**layer_idx), layers[layer_idx], stride=(2,1,1 ) ))

        # if bn_tag:
        #     self.flow.add_module('bn_last', nn.BatchNorm3d(self.inplanes))


        self.flow.add_module('last', nn.Sequential(self._make_layer(block, self.inplanes, 1, stride=(16,1,1))))
        if last_layer_softmax and layer_idx + 1 == self.layer_num:
            self.flow.add_module('last_layer', conv3x3x3(self.inplanes, self.inplanes))

        # if self.last_layer_softmax:
        #     self.proj = nn.Sequential()
        #     self.proj.add_module('last_layer', conv3x3x3(self.inplanes,self.inplanes))
        #     self.proj.add_module('Relu', nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.flow(x)


        x = x.permute(0, 2, 1, 3, 4)  # .contiguous()
        x = x.squeeze(dim=1)

        if self.last_layer_softmax:
            x = self.norm(x, p=2, dim=1)
            # x = self.proj(x)
            # x = self.norm(x, p=1, dim=1)

        return x # , m




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(3, 3, 3),
                    # kernel_size=(127,3,3),
                    stride=stride,
                    padding=(1, 1, 1),
                    # padding=(63,1,1),
                    bias=False),
            )
                # nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

# def _make_layer_last(self, block, planes, blocks, stride=1):
#     downsample = None
#     if stride != 1 :
#         downsample = nn.Sequential(
#             nn.Conv3d(
#                 self.inplanes,
#                 planes * block.expansion,
#                 kernel_size=1,
#                 stride=stride,
#                 bias=False),
#         )
#         # nn.BatchNorm3d(planes * block.expansion))
#
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#         layers.append(block(self.inplanes, planes))
#
#     return nn.Sequential(*layers)


if __name__ == "__main__":
    y = Conv3DSimple(32, 64)
    x = torch.randn(4, 1, 256,32,32)

    print(y)
    model = Res3D_net(res3dblocks, 1, last_layer_softmax=True)


    f = model(x)
    print(model)


