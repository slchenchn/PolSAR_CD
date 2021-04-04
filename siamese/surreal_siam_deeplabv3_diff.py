'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-03
'''
# 复数孪生deeplabv3网路

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from complexPytorch import *
from ptsemseg.models.resnet import *
from ptsemseg.models.complex_aspp import *
from ptsemseg.models.aspp import *


class surreal_siam_deeplabv3_diff(nn.Module):
    def __init__(self, input_nbr, num_classes, channel_scale=0.25, pretrained=True):
        super().__init__()
        self.backbone = surreal_ResNet18_OS8(pretrained=pretrained, input_nbr=input_nbr, channel_scale=channel_scale)
        self.aspp = ASPP(num_classes=num_classes, channel_scale=channel_scale)

    def forward(self, x1, x2):
        h, w = x1.shape[-2:]

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        diff = torch.abs(x1-x2)

        output = self.aspp(diff)

        output = F.upsample(output, size=(h, w), mode='bilinear')

        return output

