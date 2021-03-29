'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-03-27
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from surReal.layers import *

class surReal_SiamUnet_diff3(nn.Module):
    '''surRealed SiamUnet_diff segmentation network
    2021-03-19 用surRseal的卷积层来替换原本的实数卷积层，仅替换stage 1
    不使用 padding 参数
    2020-03-25 增加一个复数卷积层
 '''

    def __init__(self, input_nbr, label_nbr, drop_p=0.2):
        super().__init__()

        self.input_nbr = input_nbr

        # surReal part
        self.surreal_conv1 = ComplexConv2Deffgroup(input_nbr, 16, kern_size=(3,3), stride=(2,2))
        self.surreal_conv2 = ComplexConv2Deffgroup(16, 16, kern_size=(3,3), stride=(2,2))
        self.surreal_conv3 = ComplexConv2Deffgroup(16, 16, kern_size=(3,3), stride=(2,2))
        self.surreal_proj2 = manifoldReLUv2angle(16)
        self.surreal_linear1 = ComplexLinearangle2Dmw_outfield(16*63**2)
        self.upconv0 = nn.Upsample(size=(512, 512), mode='bilinear')
        # self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        # self.bn11 = nn.BatchNorm2d(16)
        # self.do11 = nn.Dropout2d(drop_p)
        # self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.bn12 = nn.BatchNorm2d(16)
        # self.do12 = nn.Dropout2d(drop_p)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(drop_p)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(drop_p)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(drop_p)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(drop_p)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(drop_p)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(drop_p)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(drop_p)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(drop_p)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv43d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(drop_p)
        self.conv42d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(drop_p)
        self.conv41d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(drop_p)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv33d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(drop_p)
        self.conv32d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(drop_p)
        self.conv31d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(drop_p)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv22d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(drop_p)
        self.conv21d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(drop_p)

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv12d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(drop_p)
        self.conv11d = nn.Conv2d(16, label_nbr, kernel_size=3, padding=1)
   
    def forward(self, x1, x2):
        # stage 1
        x11 = self.surreal_conv1(x1)
        x11 = self.surreal_proj2(x11)
        x11 = self.surreal_conv2(x11)
        x11 = self.surreal_proj2(x11)
        x11 = self.surreal_conv3(x11)
        x11 = self.surreal_proj2(x11)
        x12_1 = self.upconv0(self.surreal_linear1(x11))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # torch.norm(self.surreal_linear1(x11)).backward()
        # torch.norm(x12_1).backward()/dsf
        # torch.norm(x1p).backward()
        
        # Stage 1
        # x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        # x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        # x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.surreal_conv1(x2)
        x11 = self.surreal_proj2(x11)
        x11 = self.surreal_conv2(x11)
        x11 = self.surreal_proj2(x11)
        x11 = self.surreal_conv3(x11)
        x11 = self.surreal_proj2(x11)
        x12_2 = self.upconv0(self.surreal_linear1(x11))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # stage 1
        # x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        # x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        # x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)



        # Stage 4d
        x4d = self.upconv4(x4p)
        x4d = torch.cat((x4d, torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        x3d = torch.cat((x3d, torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        x2d = torch.cat((x2d, torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        x1d = torch.cat((x1d, torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d
        # return self.sm(x11d)

