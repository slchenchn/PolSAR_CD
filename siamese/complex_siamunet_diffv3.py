'''
Author: Shuailin Chen
Created Date: 2021-03-12
Last Modified: 2021-03-31
	content: 
'''
# 2020-10-16 将softmax层移除，因为ptorch里面的计算交叉熵的函数内部已经集成了softmax
# 统一设置所有的 drop out 概率
# 将所有的网络层改为复数, batchnorm 先设置为 naive 形式的，并将最后的输出设置为原输出的模，因为这样才能用交叉熵
# 如果将复数张量和实数张量进行 cat，后续的自动求导会出问题，最新的pytorch已经解决了，但是我用的这个版本还没
# 将反卷积改成双线性上采样
# 在网络的最开始添加 BN 层
# 将BN层从naive改成complex

# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# import torch
import torch.nn as nn
import torchvision.models
# import torch.nn.functional as F
# from torch.nn.modules.padding import ComplexReplicationPad2d
from complexPytorch import *

class complex_SiamUnet_diffv3(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, label_nbr, drop_p=0.2, channel_scale=0.5):
        super().__init__()

        self.input_nbr = input_nbr

        # self.bn0 = ComplexBatchNorm2d(input_nbr)

        self.conv11 = ComplexConv2d(input_nbr, int(16*channel_scale), kernel_size=3, padding=1)
        self.bn11 = ComplexBatchNorm2d(int(16*channel_scale))
        self.do11 = ComplexDropout2d(drop_p)
        self.conv12 = ComplexConv2d(int(16*channel_scale), int(16*channel_scale), kernel_size=3, padding=1)
        self.bn12 = ComplexBatchNorm2d(int(16*channel_scale))
        self.do12 = ComplexDropout2d(drop_p)

        self.conv21 = ComplexConv2d(int(16*channel_scale), int(32*channel_scale), kernel_size=3, padding=1)
        self.bn21 = ComplexBatchNorm2d(int(32*channel_scale))
        self.do21 = ComplexDropout2d(drop_p)
        self.conv22 = ComplexConv2d(int(32*channel_scale), int(32*channel_scale), kernel_size=3, padding=1)
        self.bn22 = ComplexBatchNorm2d(int(32*channel_scale))
        self.do22 = ComplexDropout2d(drop_p)

        self.conv31 = ComplexConv2d(int(32*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn31 = ComplexBatchNorm2d(int(64*channel_scale))
        self.do31 = ComplexDropout2d(drop_p)
        self.conv32 = ComplexConv2d(int(64*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn32 = ComplexBatchNorm2d(int(64*channel_scale))
        self.do32 = ComplexDropout2d(drop_p)
        self.conv33 = ComplexConv2d(int(64*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn33 = ComplexBatchNorm2d(int(64*channel_scale))
        self.do33 = ComplexDropout2d(drop_p)

        self.conv41 = ComplexConv2d(int(64*channel_scale), int(128*channel_scale), kernel_size=3, padding=1)
        self.bn41 = ComplexBatchNorm2d(int(128*channel_scale))
        self.do41 = ComplexDropout2d(drop_p)
        self.conv42 = ComplexConv2d(int(128*channel_scale), int(128*channel_scale), kernel_size=3, padding=1)
        self.bn42 = ComplexBatchNorm2d(int(128*channel_scale))
        self.do42 = ComplexDropout2d(drop_p)
        self.conv43 = ComplexConv2d(int(128*channel_scale), int(128*channel_scale), kernel_size=3, padding=1)
        self.bn43 = ComplexBatchNorm2d(int(128*channel_scale))
        self.do43 = ComplexDropout2d(drop_p)

        self.upconv4 = ComplexUpsample(scale_factor=2, mode='bilinear')
        self.conv43d = ComplexConv2d(int(256*channel_scale), int(128*channel_scale), kernel_size=3, padding=1)
        self.bn43d = ComplexBatchNorm2d(int(128*channel_scale))
        self.do43d = ComplexDropout2d(drop_p)
        self.conv42d = ComplexConv2d(int(128*channel_scale), int(128*channel_scale), kernel_size=3, padding=1)
        self.bn42d = ComplexBatchNorm2d(int(128*channel_scale))
        self.do42d = ComplexDropout2d(drop_p)
        self.conv41d = ComplexConv2d(int(128*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn41d = ComplexBatchNorm2d(int(64*channel_scale))
        self.do41d = ComplexDropout2d(drop_p)

        self.upconv3 = ComplexUpsample(scale_factor=2, mode='bilinear')
        self.conv33d = ComplexConv2d(int(128*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn33d = ComplexBatchNorm2d(int(64*channel_scale))
        self.do33d = ComplexDropout2d(drop_p)
        self.conv32d = ComplexConv2d(int(64*channel_scale), int(64*channel_scale), kernel_size=3, padding=1)
        self.bn32d = ComplexBatchNorm2d(int(64*channel_scale))
        self.do32d = ComplexDropout2d(drop_p)
        self.conv31d = ComplexConv2d(int(64*channel_scale), int(32*channel_scale), kernel_size=3, padding=1)
        self.bn31d = ComplexBatchNorm2d(int(32*channel_scale))
        self.do31d = ComplexDropout2d(drop_p)

        self.upconv2 = ComplexUpsample(scale_factor=2, mode='bilinear')
        self.conv22d = ComplexConv2d(int(64*channel_scale), int(32*channel_scale), kernel_size=3, padding=1)
        self.bn22d = ComplexBatchNorm2d(int(32*channel_scale))
        self.do22d = ComplexDropout2d(drop_p)
        self.conv21d = ComplexConv2d(int(32*channel_scale), int(16*channel_scale), kernel_size=3, padding=1)
        self.bn21d = ComplexBatchNorm2d(int(16*channel_scale))
        self.do21d = ComplexDropout2d(drop_p)

        self.upconv1 = ComplexUpsample(scale_factor=2, mode='bilinear')
        self.conv12d = ComplexConv2d(int(32*channel_scale), int(16*channel_scale), kernel_size=3, padding=1)
        self.bn12d = ComplexBatchNorm2d(int(16*channel_scale))
        self.do12d = ComplexDropout2d(drop_p)
        self.conv11d = ComplexConv2d(int(16*channel_scale), label_nbr, kernel_size=3, padding=1)

        # self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):


        """Forward method."""
        # 加上一个 BN 层
        # start = self.bn0(torch.cat((x1, x2), dim=0))
        # x1 = start[:x1.shape[0], :, :, :]
        # x2 = start[x1.shape[0]:, :, :, :]

        # Stage 1
        x11 = self.do11(complex_relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(complex_relu(self.bn12(self.conv12(x11))))
        x1p = complex_avg_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(complex_relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(complex_relu(self.bn22(self.conv22(x21))))
        x2p = complex_avg_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(complex_relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(complex_relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(complex_relu(self.bn33(self.conv33(x32))))
        x3p = complex_avg_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(complex_relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(complex_relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(complex_relu(self.bn43(self.conv43(x42))))
        x4p = complex_avg_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(complex_relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(complex_relu(self.bn12(self.conv12(x11))))
        x1p = complex_avg_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(complex_relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(complex_relu(self.bn22(self.conv22(x21))))
        x2p = complex_avg_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(complex_relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(complex_relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(complex_relu(self.bn33(self.conv33(x32))))
        x3p = complex_avg_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(complex_relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(complex_relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(complex_relu(self.bn43(self.conv43(x42))))
        x4p = complex_avg_pool2d(x43_2, kernel_size=2, stride=2)

        # Stage 4d
        x4d = self.upconv4(x4p)
        x4d = torch.cat((x4d, torch.abs(x43_1 - x43_2).type(torch.complex64)), 1)
        # torch.norm(x4d.abs()).backward()
        x43d = self.do43d(complex_relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(complex_relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(complex_relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        x3d = torch.cat((x3d, torch.abs(x33_1 - x33_2).type(torch.complex64)), 1)
        x33d = self.do33d(complex_relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(complex_relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(complex_relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        x2d = torch.cat((x2d, torch.abs(x22_1 - x22_2).type(torch.complex64)), 1)
        x22d = self.do22d(complex_relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(complex_relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        x1d = torch.cat((x1d, torch.abs(x12_1 - x12_2).type(torch.complex64)), 1)
        x12d = self.do12d(complex_relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d.real
        # return self.sm(x11d)

