import torch
import torch.nn as nn
from Basicblocks import res2dblocks, conv3x3
import torch.nn.functional as F


class RUEncoder(nn.Module):
    def __init__(self, block, input_channels=3, layers=[1, 1, 1], layer_num=3, neck_planes=32, last_layer_softmax=True):
        super(RUEncoder, self).__init__()
        self.inplanes = neck_planes
        self.neck_planes = neck_planes * 2
        self.layer_num = layer_num
        self.last_layer_softmax = last_layer_softmax

        self.norm = F.normalize
        self.layer_list = nn.ModuleList()

        self.layer0 = []
        self.layer0.append(
            nn.Conv2d(input_channels, self.inplanes, kernel_size=3, padding=1, stride=1, bias=False))  # (5*3) -> 32
        self.layer0.append(nn.ReLU(inplace=True))

        self.layer_list.append(nn.Sequential(*self.layer0))

        for layer_idx in range(self.layer_num):
            temp_flow = nn.Sequential()
            temp_flow.add_module('enc_block{}'.format(layer_idx),
                                 self._make_layer(block, self.neck_planes * (2 ** layer_idx), layers[layer_idx],
                                                  stride=2))  #
            if self.last_layer_softmax and layer_idx + 1 == self.layer_num:
                temp_flow.add_module('last_layer', conv3x3(self.inplanes, self.inplanes))
            self.layer_list.append(temp_flow)

    def forward(self, x):
        out_list = []
        x = self.layer_list[0](x)  # 32*256*256
        for idx in range(1, len(self.layer_list)):
            x = self.layer_list[idx](x)
            out_list.append(x)
        if self.last_layer_softmax:
            out_list[-1] = self.norm(out_list[-1], p=1, dim=1)

        out_list = out_list[::-1]
        return out_list

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3,
                          stride=stride, padding=1, bias=False),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class RUDecoder(nn.Module):
    def __init__(self, block, output_channels=3, layers=[1, 1, 1, 1], layer_num=3, neck_planes=32, tanh_tag=False):
        super(RUDecoder, self).__init__()
        self.layer_num = layer_num
        self.neck_planes = neck_planes * 2
        self.inplanes = self.neck_planes * (2 ** (self.layer_num))

        self.layer_list = nn.ModuleList()

        self.layer0 = []
        self.layer0.append(nn.Conv2d(self.neck_planes//2, output_channels, kernel_size=3, stride=1, padding=1, bias=False))

        layer_channel_num = [256, 128, 64, 64]  # [512, 256, 128, 64]

        layers_idx = 0
        self.layer_list.append(
            self._make_layer(block, self.neck_planes * (2 ** (self.layer_num - 2 - layers_idx)), layers[layers_idx],
                             stride=2))

        for layers_idx in range(1, self.layer_num - 1):
            self.layer_list.append(
                self._make_layer(block, self.neck_planes * (2 ** (self.layer_num - 2 - layers_idx)), layers[layers_idx],
                                 stride=2, cat_tag=2))

        self.layer_list.append(
            self._make_layer(block, int(self.neck_planes * (2 ** (self.layer_num - 2 - layers_idx - 1))), layers[layers_idx + 1],
                             stride=2, cat_tag=2))

        self.layer_list.append(nn.Sequential(*self.layer0))

    def forward(self, x_list):
        x = x_list[0]
        x = self.layer_list[0](x)
        for idx in range(1, len(x_list)):
            x = torch.cat([x, x_list[idx]], 1)
            x = self.layer_list[idx](x)

        x = self.layer_list[-1](x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, cat_tag=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = [
                nn.PixelShuffle(upscale_factor=stride),
                nn.Conv2d(self.inplanes * cat_tag // (stride ** 2), planes,
                          kernel_size=3, stride=1, padding=1, bias=False),
            ]
            downsample = nn.Sequential(*downsample)
        layers = [block(self.inplanes * cat_tag, planes, -1 * stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class UResAE(nn.Module):
    def __init__(self, video_channels_in=3, video_channels_out=3, encoder_layers=[1, 1, 1, 1],
                 decoder_layers=[1, 1, 1, 1], layer_num=3, neck_planes=32):
        super(UResAE, self).__init__()
        self.encoder = RUEncoder(res2dblocks, input_channels=video_channels_in, layers=encoder_layers)
        self.decoder = RUDecoder(res2dblocks, output_channels=video_channels_out, layers=decoder_layers)

    def forward(self, x, y):
        x = self.encoder(x)
        out_encoder = x
        out_encoder[0] = torch.cat([out_encoder[0],y], dim=1)
        z = self.decoder(out_encoder)
        out_decoder = z
        return out_encoder, out_decoder


if __name__ == "__main__":
    model = UResAE(4, 3)
    x = torch.rand(2, 4, 256, 256)
    y = torch.randn(2, 256, 32, 32)
    out = model(x, y)
    print(out[1].shape)
