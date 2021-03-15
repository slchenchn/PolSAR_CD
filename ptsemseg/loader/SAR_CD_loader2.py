''' version 2 '''

import argparse
from operator import imod, index
from re import search
from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import os
import os.path as osp
import sys
from typing import Union
from mylib import polSAR_utils as psr
from mylib import labelme_utils as lbm
import re
from matplotlib import pyplot as plt 
import numpy as np
import cv2

class SAR_CD_base(data.Dataset):
    ''' the base class of the PolSAR change detection dataloaders '''
    def __init__(self, root,        
        split="train",
        transform=None,
        # img_size=512,
        augments=None,
        to_tensor = True,
        data_type = 'pauli'
        # n_classes = 7
        # data_format = 'save_space'
        ):
        super().__init__()
        self.root = root
        self.split = split
        # self.data_format = data_format
        self.augments = augments
        self.transfrom = transform
        self.n_classes = 2
        self.to_tensor = to_tensor
        self.data_type = data_type
        print('data type:', data_type)

        # read change label images' path
        self.labels_path = []
        for super_dir, _, files in os.walk(self.root):
            for file in files:
                if '-change.png' in file:
                    self.labels_path.append(osp.join(super_dir, file))

        if self.split == 'train':
            self.labels_path = self.labels_path[0: int(len(self.labels_path)*0.8)]
        elif self.split == 'val':
            self.labels_path = self.labels_path[int(len(self.labels_path)*0.8): ]

        # transform
        if self.to_tensor:
            self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.labels_path)

    def get_label_and_mask(self, index:int):
        ''' generate label and its mask, in torch.tensor forat '''
        label_path = self.labels_path[index]
        label = lbm.read_change_label_png(label_path)-1
        mask = label<2      # 1 表示存在有效标记，0表示没有标记
        label[~mask] = 0    # 1 表示存在变化，0表示没有变化或没有标记数据
        return torch.from_numpy(label).long(), torch.from_numpy(mask)

    def get_files_data(self, index):
        ''' read file a and file b data, in torch.tensor format
        if the data is images, then it is normed into [0,1 ], 
        '''
        label_path = self.labels_path[index]
        # get the file path
        label_dir = osp.split(label_path)[0]
        # two date time display format
        if osp.isfile(osp.join(label_dir, 'pin.txt')):   # mode 1
            re_exp = r'20\d{6}'
        elif osp.isfile(osp.join(label_dir, 'pin-2.txt')):   # mode 2
            re_exp = r'\d{2}[A-Z][a-z]{2}\d{4}'
        else:
            raise ValueError('unknown mode of label format')
        [a_time, b_time] = re.findall(re_exp, label_path)

        files_path = []
        files_path.append(osp.join(label_dir.replace('label', 'data'), a_time, 'C3'))
        files_path.append(osp.join(label_dir.replace('label', 'data'), b_time, 'C3'))

        # get the file data
        slice_idx = re.search(r'-\d{4}-', label_path)
        if slice_idx is None:
            raise ValueError('can not find the wave code')
        slice_idx = int(slice_idx.group()[1:5])

        files = []
        if self.data_type=='original':
            for ii in range(2):
                files.append(torch.from_numpy(psr.read_c3(osp.join(files_path[ii], str(slice_idx)), out=self.data_format)))
        elif self.data_type=='pauli':
            for ii in range(2):
                files.append(psr.read_bmp(osp.join(files_path[ii], str(slice_idx))))
            if self.to_tensor:
                for ii in range(2):
                    files[ii] = self.tf(files[ii])
            else:
                for ii in range(2):
                    files[ii] = files[ii].permute(2, 0, 1)
        else:
            raise NotImplementedError
        return files

    def statistics(self):
        ''' calculate the statistics of the labels'''
        # absolute value
        cnt_change = 0
        cnt_unchange = 0
        cnt_unlabeled = 0
        for label_path in self.labels_path:
            label = lbm.read_change_label_png(label_path)
            cnt_unlabeled += np.count_nonzero(label==0)
            cnt_unchange += np.count_nonzero(label==1)
            cnt_change += np.count_nonzero(label==2)
            
        # percentage
        cnt_all = cnt_change + cnt_unchange + cnt_unlabeled
        if cnt_all!=len(self)*512**2:
            print('cnt_all wrong, cnt_all=', cnt_all, ', actually it should be', len(self)*512**2)
        pec_change = cnt_change / cnt_all
        pec_unchange = cnt_unchange / cnt_all
        per_unlabeled = cnt_unlabeled / cnt_all

        # prints
        print(f'changed: {cnt_change}, {pec_change*100}\%')
        print(f'unchanged: {cnt_unchange}, {pec_unchange*100}\%')
        print(f'unlabeled: {cnt_unlabeled}, {per_unlabeled*100}\%')
        print('unchanged / changed:', cnt_unchange/cnt_change)


class SAR_CD_semseg(SAR_CD_base):
    ''' change detection using semantic segmentatios's method '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)   

    def __getitem__(self, index):
        


            

if __name__ == '__main__':
    print('pwd: ', os.getcwd())
    # print('sys path: ', sys.path)
    print('_'*20, '\n')
    
    opt = dict()
    opt['dataset_path'] = './data/SAR_CD/RS2'
    opt['data_format'] = 'save_space'
    # opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # opt.add_argument('--dataset path', type=str, )
    
    
    ''' SAR_CD_tile_1 '''
    dataset = SAR_CD_tile_1(root=opt['dataset_path'], split='train', tile_size=32)
    print('size of dataset: ', len(dataset))
    file_a, file_b, label = dataset.__getitem__(4586)
    save_dir = './tmp/'
    # file_a = ff['file_a']
    # file_b = ff['file_b']
    # for ii in range(9):
    #     plt.hist(file_a[ii, :, :])
    #     plt.savefig(osp.join(save_dir, f'a-{ii}.png'))
    #     plt.hist(file_b[ii, :, :])
    #     plt.savefig(osp.join(save_dir, f'b-{ii}.png'))
    #     # cv2.imwrite(osp.join(save_dir, f'a-{ii}.png'), 255*file_a[ii, :, :])
    #     # cv2.imwrite(osp.join(save_dir, f'b-{ii}.png'), 255*file_b[ii, :, :])

    ''' SAR_CD_Hoekman '''
    # dataset = SAR_CD_Hoekman(root=opt['dataset_path'], split='test')
    # print('size of dataset: ', len(dataset))
    # ff = dataset.__getitem__(5)
    # save_dir = './tmp/'
    # file_a = ff['file_a']
    # file_b = ff['file_b']
    # for ii in range(9):
    #     plt.hist(file_a[ii, :, :])
    #     plt.savefig(osp.join(save_dir, f'a-{ii}.png'))
    #     plt.hist(file_b[ii, :, :])
    #     plt.savefig(osp.join(save_dir, f'b-{ii}.png'))
    #     # cv2.imwrite(osp.join(save_dir, f'a-{ii}.png'), 255*file_a[ii, :, :])
    #     # cv2.imwrite(osp.join(save_dir, f'b-{ii}.png'), 255*file_b[ii, :, :])
        
    # plt.figure()
    # plt.imshow(psr.rgb_by_c3(ff['file_a']))
    # plt.savefig(save_dir+'a.png')
    # plt.figure()
    # tmp = ff['file_a']
    # plt.imshow(psr.rgb_by_c3())
    # plt.savefig(save_dir+'a.png')
    # print('path_a: ', ff['path_a'])
    # plt.figure()
    # plt.imshow(psr.rgb_by_c3(ff['file_b']))
    # plt.savefig(save_dir+'b.png')
    # print('path_b ', ff['path_b'])

    # print('label path: ', ff['path_label'])
    # plt.figure()
    # plt.imshow(ff['label'])
    # plt.savefig(save_dir+'label.png')
    # plt.figure()
    # plt.imshow(ff['mask'])
    # plt.savefig(save_dir+'mask.png')

    # ''' SAR_CD_direct '''
    # dataset = SAR_CD_direct(root=opt['dataset_path'], split='train')
    # print('size of dataset: ', len(dataset))
    # ff = dataset.__getitem__(20)
    # save_dir = './tmp/'
    # plt.figure()
    # plt.imshow(psr.rgb_by_c3(ff['file_a']))
    # plt.savefig(save_dir+'a.png')
    # plt.figure()
    # tmp = ff['file_a']
    # plt.imshow(psr.rgb_by_c3())
    # plt.savefig(save_dir+'a.png')
    # print('path_a: ', ff['path_a'])
    # plt.figure()
    # plt.imshow(psr.rgb_by_c3(ff['file_b']))
    # plt.savefig(save_dir+'b.png')
    # print('path_b ', ff['path_b'])

    # print('label path: ', ff['path_label'])
    # plt.figure()
    # plt.imshow(ff['label'])
    # plt.savefig(save_dir+'label.png')
    # plt.figure()
    # plt.imshow(ff['mask'])
    # plt.savefig(save_dir+'mask.png')

    # clrmap = np.array([[0,0,0],[255, 255, 255],[0,255,0]])
    # lbm.lblsave(save_dir+'label.png', ff['label'], clrmap)
    # lbm.lblsave(save_dir+'mask.png', ff['mask'], clrmap)
    # print('label path:', ff['path_label'])


    ''' SAR_CD_intersity '''
    # dataset = SAR_CD_intensities(root=opt['dataset_path'], split='train')
    # print('size of dataset: ', len(dataset))

    # print('train set:')
    # for item in dataset.labels_path:
    #     print(item)
    # print('_'*100)
    # dataset = SAR_CD_intensities(root=opt['dataset_path'], split='test')
    # print('val set:')
    # for item in dataset.labels_path:
    #     print(item)

    # ff = dataset.__getitem__(5)
    # save_dir = './tmp/'
    # file_a = ff['file_a'].permute(1,2,0).numpy()
    # print(type(file_a), file_a.dtype, file_a.shape)
    # cv2.imwrite(save_dir+'a.png', file_a)
    # file_b = ff['file_b'].permute(1,2,0).numpy()
    # cv2.imwrite(save_dir+'b.png', file_b)
    # # plt.figure()
    # # plt.imshow(psr.rgb_by_c3(ff['file_a']))
    # # # plt.show()
    # # plt.savefig(save_dir+'a.png')
    # # plt.figure()
    # # plt.imshow(psr.rgb_by_c3(ff['file_b']))
    # # # plt.show()
    # # plt.savefig(save_dir+'b.png')

    # clrmap = np.array([[0,0,0],[255, 255, 255],[0,255,0]])
    # lbm.lblsave(save_dir+'label.png', ff['label'], clrmap)
    # lbm.lblsave(save_dir+'mask.png', ff['mask'], clrmap)
    # print('label path:', ff['path_label'])

    ''' calculate statistics '''
    # dataset = SAR_CD_base(root=opt['dataset_path'], split='train')
    # print('size of dataset: ', len(dataset))
    # dataset.statistics()

    
