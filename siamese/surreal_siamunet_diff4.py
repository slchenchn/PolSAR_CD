'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-02
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from surReal.shrinkage_layers import *
from surReal.layers import ComplexLinearangle2Dmw_outfield

class surReal_SiamUnet_diff4(nn.Module):
    '''全新设计的复数差分孪生网络，用c-sure的代码
    失败了，需要的参数量太大，训练时间也非常长
 '''

    def __init__(self, input_nbr, label_nbr, drop_p=0.2):
        super().__init__()

        self.input_nbr = input_nbr

        self.conv11 = ComplexConv2Deffangle4Dxy(input_nbr, 8, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv12 = ComplexConv2Deffangle4Dxy(8, 8, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj11 = ReLU4Dsp(8)
        self.proj12 = ReLU4Dsp(8)
        self.dist1 = ComplexLinearangle2Dmw_outfield(8*512**2)

        self.conv21 = ComplexConv2Deffangle4Dxy(8, 16, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv22 = ComplexConv2Deffangle4Dxy(16, 16, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj21 = ReLU4Dsp(16)
        self.proj22 = ReLU4Dsp(16)
        self.dist2 = ComplexLinearangle2Dmw_outfield(16*256**2)

        self.conv31 = ComplexConv2Deffangle4Dxy(16, 32, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv32 = ComplexConv2Deffangle4Dxy(32, 32, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj31 = ReLU4Dsp(32)
        self.proj32 = ReLU4Dsp(32)
        self.dist3 = ComplexLinearangle2Dmw_outfield(32*128**2)

        self.conv41 = ComplexConv2Deffangle4Dxy(32, 64, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv42 = ComplexConv2Deffangle4Dxy(64, 64, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj41 = ReLU4Dsp(64)
        self.proj42 = ReLU4Dsp(64)
        self.dist4 = ComplexLinearangle2Dmw_outfield(64*64**2)

        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv42d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(32)
        self.bn41d = nn.BatchNorm2d(16)

        self.conv32d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(16)
        self.bn31d = nn.BatchNorm2d(8)
        
        self.conv22d = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(8)
        self.bn21d = nn.BatchNorm2d(4)
        
        self.conv12d = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv11d = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(4)
       
   
    def forward(self, x1, x2):
        x1
        x11 = self.proj11(self.conv11(x1))
        x11 = self.proj12(self.conv12(x11))
        x12_1 = self.dist1(x11)
        
        x21 = self.proj21(self.conv21(x11))
        # delete
        x21 = self.proj22(self.conv22(x21))
        x22_1 = self.dist2(x21)
        
        x31 = self.proj31(self.conv31(x21))
        x31 = self.proj32(self.conv32(x31))
        x32_1 = self.dist3(x31)
        
        x41 = self.proj41(self.conv41(x31))
        x41 = self.proj42(self.conv42(x41))
        x42_1 = self.dist4(x41)

        ##########################################
        x11 = self.proj11(self.conv11(x2))
        x11 = self.proj12(self.conv12(x11))
        x12_2 = self.dist1(x11)
        
        x21 = self.proj21(self.conv21(x11))
        x21 = self.proj22(self.conv22(x21))
        x22_2 = self.dist2(x21)
        
        x31 = self.proj31(self.conv31(x21))
        x31 = self.proj32(self.conv32(x31))
        x32_2 = self.dist3(x31)
        
        x41 = self.proj41(self.conv41(x31))
        x41 = self.proj42(self.conv42(x41))
        x42_2 = self.dist4(x41)

        ###########################################
        x4d = torch.abs(x42_1-x42_2)
        x42d = F.relu(self.bn42d(self.conv42d(x4d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x41d = self.upconv(x41d)

        x3d = torch.cat(x41d, torch.abs(x32_1-x32_2), dim=2)
        x32d = F.relu(self.bn32d(self.conv32d(x3d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x31d = self.upconv(x31d)
        
        x2d = torch.cat(x31d, torch.abs(x22_1-x22_2), dim=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x21d = self.upconv(x21d)
        
        x1d = torch.cat(x21d, torch.abs(x12_1-x12_2), dim=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = F.relu(self.bn11d(self.conv11d(x12d)))
        x11d = self.upconv(x11d)

        return x11d