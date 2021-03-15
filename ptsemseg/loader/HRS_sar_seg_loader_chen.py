# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/4
# @Author  : Chenda
# 梁烽的写的 dataloader 没有做，因此我做一个对数变换

from io import FileIO
import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import *

import cv2 as cv
from torchvision import transforms
import os.path as osp

class HRS_SAR_seg_chen_Loader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,      #normalization
        n_classes = 7
    ):
        self.root = root
        self.split = split
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = n_classes
        self.files = collections.defaultdict(list)

        # 20% val, 80% train
        file_list = os.listdir(self.root)
        gt_file_list = []
        for fs in file_list:
            if 'gt' in fs:
                gt_file_list.append(fs)
        file_list = gt_file_list
        if split == 'train':
            self.files[self.split] = file_list[0: int(len(file_list)*0.8)]
        else:
            self.files[self.split] = file_list[int(len(file_list)*0.8): ]     

        # for split in ["train", "test", "val"]:
        #     file_list = os.listdir(osp.join(root, split))
        #     self.files[split] = file_list

        # self.tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.45222182, 0.32558659, 0.32138991],
        #                                                    [0.21074223, 0.14708663, 0.14242824])])
        # self.tf_no_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.45222182, 0.32558659, 0.32138991],
        #                                                                           [1,1,1])])
        self.tf = transforms.ToTensor()
        self.tf_no_train = transforms.ToTensor()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        lbl_path = osp.join(self.root, img_name)
        lbl = cv.imread(lbl_path, -1)
        return lbl
        # if self.augmentations is not None:
        #     img, lbl = self.augmentations(img, lbl)

        # if self.is_transform:
        #     img, lbl = self.transform(img, lbl)

        # return img, lbl

    def transform(self, img, lbl):
        ''' resize, transform to tensor '''
        if self.img_size == ('same', 'same'):
            pass
        else:
            #opencv resize,(width,heigh)
            # try:
            img=cv.resize(img,(self.img_size[1],self.img_size[0]))
            # except:
                # print('error breakpoint')
            lbl = cv.resize(lbl, (self.img_size[1], self.img_size[0]))

            # img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            # lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        if self.split=="train":
            img = self.tf(img)
        else:
            img=self.tf_no_train(img)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):   # 这里是BGR值，因为是用cv2读取的数据
        class1=[0,0,0]              # 黑色， 无数据
        class2=[255, 255, 0]        # 浅蓝色，水体
        class3 = [0, 255, 255]      # 黄色，建筑物
        class4 = [255, 255, 255]    # 白色，其他
        class5 = [0, 0, 255]        # 红色，土地、裸地
        class6=[255, 0, 0]          # 蓝色，工业区
        class7=[0, 255, 0]          # 绿色，林草地

        label_colours = np.array(
            [
                class1,
                class2,
                class3,
                class4,
                class5,
                class6,
                class7
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb
    


