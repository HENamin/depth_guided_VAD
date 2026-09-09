import numpy as np
import torch
import torch.nn as nn

from config import seed
RANDOM_SEED = seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)





def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)


def conv_up3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    downsample = [
        nn.PixelShuffle(upscale_factor=stride),
        nn.Conv2d(in_planes//(stride**2), out_planes,kernel_size=3, stride=1,padding=1, bias=False),
    ]
    downsample = nn.Sequential(*downsample)

    return downsample

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3,3,3),
        # kernel_size=(127,3,3),
        stride=stride,
        padding=(1,1,1),
        # padding=(63,1,1),
        bias=False)

class res2dblocks(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(res2dblocks, self).__init__()

        if stride == -2:
            self.conv1 = conv_up3x3(inplanes, planes, -1*stride)
        # import pdb;pdb.set_trace()
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)


        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride  # why?

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class res3dblocks(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(res3dblocks, self).__init__()

        # if stride == -2:
        #     self.conv1 = conv_up3x3(inplanes, planes, -1*stride)
        # # import pdb;pdb.set_trace()
        # else:
        self.conv1 = conv3x3x3(inplanes, planes, stride)

        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride  # why?

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# if __name__ == "__main__":
#     # a = res2dblocks(32,64,stride=2, downsample=None   )
#     # print (a)


#     a1 = res3dblocks(32,64,stride=2, downsample=None   )
#     print(a1)

#     x = torch.randn(1, 32, 256,32,32)
#     c = a1(x)