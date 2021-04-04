'''
Author: Shuailin Chen
Created Date: 2021-03-11
Last Modified: 2021-04-03
	content: 
'''
''' 适用于变化检测的数据增广方式 '''

import math
import numbers
import random
import numpy as np
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torchvision.transforms.functional_tensor as F_t
import torch
from PIL import Image, ImageOps
from torch import nn
import torch.nn.functional as F


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, file_a, file_b, label, mask):
        # file_a = Image.fromarray(file_a, mode="F")
        # file_b = Image.fromarray(file_b, mode="F")
        # label = Image.fromarray(label, mode="L")
        # mask = Image.fromarray(mask, mode="L")

        for a in self.augmentations:
            file_a, file_b, label, mask = a(file_a, file_b, label, mask)

        # file_a, file_b, label, mask = torch.from_numpy(file_a), torch.from_numpy(file_b), torch.from_numpy(label), torch.from_numpy(mask)

        return file_a, file_b, label, mask

 

class Boxcar_smooth(object):
    def __init__(self, kernel_size=3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.ones(kernel_size, kernel_size)/(kernel_size**2)
        self.padding = int((kernel_size-1)/2)

    def __call__(self, file):
        return F.conv2d(file, self.kernel, bias=None, padding=self.padding)



class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file_a, file_b, label, mask):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return (
                torch.flip(file_a, (-1,)),
                torch.flip(file_b, (-1,)),
                torch.flip(label, (-1,)),
                torch.flip(mask, (-1,)),
                # file_a.transpose(Image.FLIP_LEFT_RIGHT),
                # file_b.transpose(Image.FLIP_LEFT_RIGHT),
                # label.transpose(Image.FLIP_LEFT_RIGHT),
                # mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return file_a, file_b, label, mask

        
class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file_a, file_b, label, mask):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return (
                torch.flip(file_a, (-2,)),
                torch.flip(file_b, (-2,)),
                torch.flip(label, (-2,)),
                torch.flip(mask, (-2,)),
                # file_a.transpose(Image.FLIP_LEFT_RIGHT),
                # file_b.transpose(Image.FLIP_LEFT_RIGHT),
                # label.transpose(Image.FLIP_LEFT_RIGHT),
                # mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return file_a, file_b, label, mask


class RandomRotation(object):
    def __init__(self, degrees, 
    interpolation=0, # 0是最近邻，2是双线性
    expand=False, fill=None) -> None:
        super().__init__()
        self.degrees = _setup_angle(degrees)
        # self.center = center
        self.resample = interpolation
        self.expand = expand
        self.fill = fill

    def __call__(self, file_a, file_b, label, mask):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        # print('angle: ', angle)
        center_f = [0.0, 0.0]
        matrix = tf._get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
        return (
            F_t.rotate(file_a, matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill),
            F_t.rotate(file_b, matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill),
            F_t.rotate(label.unsqueeze(0), matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill).squeeze(0),
            F_t.rotate(mask.unsqueeze(0), matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill).squeeze(0)
        )
        # catted = torch.cat(file_a, file_b, label, mask, dim=)

def _setup_angle(x):
    if isinstance(x, numbers.Number):
        x = [-x, x]

    return [float(d) for d in x]



if __name__=='__main__':
    a = torch.arange(16).reshape(1, 1, 4, 4)
    f = Boxcar_smooth(3)
    b = f(a)
    print('done')