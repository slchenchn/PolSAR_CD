'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-02
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from surReal.layers import *

class surReal_SiamUnet_diff6(nn.Module):
    '''全新设计的复数差分孪生网络，用surreal的代码
    相比于v5，减少了参数量
 '''

    def __init__(self, input_nbr, label_nbr, drop_p=0.2):
        super().__init__()

        self.input_nbr = input_nbr

        self.conv11 = ComplexConv2Deffgroup(input_nbr, 8, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv12 = ComplexConv2Deffgroup(8, 8, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.proj11 = manifoldReLUv2angle(8)
        self.proj12 = manifoldReLUv2angle(8)
        self.dist1 = ComplexLinear_per_channel(8, 128)

        self.conv21 = ComplexConv2Deffgroup(8, 16, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv22 = ComplexConv2Deffgroup(16, 16, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj21 = manifoldReLUv2angle(16)
        self.proj22 = manifoldReLUv2angle(16)
        self.dist2 = ComplexLinear_per_channel(16, 64)

        self.conv31 = ComplexConv2Deffgroup(16, 32, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv32 = ComplexConv2Deffgroup(32, 32, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj31 = manifoldReLUv2angle(32)
        self.proj32 = manifoldReLUv2angle(32)
        self.dist3 = ComplexLinear_per_channel(32, 32)

        self.conv41 = ComplexConv2Deffgroup(32, 64, kern_size=(3,3), stride=(2, 2), padding=(1,1))
        self.conv42 = ComplexConv2Deffgroup(64, 64, kern_size=(3,3), stride=(1, 1), padding=(1,1))
        self.proj41 = manifoldReLUv2angle(64)
        self.proj42 = manifoldReLUv2angle(64)
        self.dist4 = ComplexLinear_per_channel(64, 16)

        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv42d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(32)
        self.bn41d = nn.BatchNorm2d(32)

        self.conv32d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(32)
        self.bn31d = nn.BatchNorm2d(16)
        
        self.conv22d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(16)
        self.bn21d = nn.BatchNorm2d(8)
        
        self.conv12d = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv11d = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(8)
       
   
    def forward(self, x1, x2):
        ''' 输入为5维向量，分别是[batch, 2, channel, height, width] 
        第二维为 [角度，幅度]，角度的范围为[0-pi] 
        '''
        x11 = self.conv11(x1)   #一个这样的卷积层怎么占了这么多的显存
        x11 = self.proj11(x11)
        x11 = self.proj12(self.conv12(x11))
        x12_1 = self.dist1(x11)
        
        x21 = self.proj21(self.conv21(x11))
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

        x3d = torch.cat((x41d, torch.abs(x32_1-x32_2)), dim=1)
        x32d = F.relu(self.bn32d(self.conv32d(x3d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x31d = self.upconv(x31d)
        
        x2d = torch.cat((x31d, torch.abs(x22_1-x22_2)), dim=1)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x21d = self.upconv(x21d)
        
        x1d = torch.cat((x21d, torch.abs(x12_1-x12_2)), dim=1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        
        x11d = self.upconv(self.upconv(x11d))

        return x11d