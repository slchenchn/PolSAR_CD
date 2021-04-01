'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-03-31
'''
# 孪生deeplabv3网路，最简单的adaption

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from complexPytorch import *
from ptsemseg.models.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from ptsemseg.models.aspp import ASPP, ASPP_Bottleneck

class siam_deeplabv3_diff(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr, num_classes, drop_p=0.2, pretrained=True):
        super().__init__()
        self.backbone = ResNet18_OS8(pretrained=pretrained, input_nbr=input_nbr)
        self.aspp = ASPP(num_classes=num_classes)

    def forward(self, x1, x2):
        h, w = x1.shape[-2:]

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        diff = torch.abs(x1-x2)

        output = self.aspp(diff)

        output = F.upsample(output, size=(h, w), mode='bilinear')

        return output

