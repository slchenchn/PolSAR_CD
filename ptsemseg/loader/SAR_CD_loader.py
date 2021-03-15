import argparse
from operator import imod, index, truediv
from re import search
import shutil
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
# sys.path.append(osp.abspath(os.getcwd()))   #很奇怪，必须用绝对路径
# print('sys path: ', sys.path)
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
        augmentations=None,
        to_tensor = True,
        data_type = 'pauli'
        # n_classes = 7
        # data_format = 'save_space'
        ):
        super().__init__()
        self.root = root
        self.split = split
        # self.data_format = data_format
        self.augmentations = augmentations
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
        # cv2.imwrite('tmp/mask.png', (mask*255).astype(np.uint8))
        # cv2.imwrite('tmp/label.png', (label*255).astype(np.uint8))
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


class SAR_CD_intensities(SAR_CD_base):
    ''' read the diagonal items of the C3 matrix as the input data
    '''
    def __init__(self, *args, time_shuffle=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)   
        self.time_shuffle = time_shuffle
        # 20% val, 80% train
        if self.split in ('train', 'test_train'):
            self.labels_path = self.labels_path[0: int(len(self.labels_path)*0.8)]
        elif self.split in ('val', 'test_val'):
            self.labels_path = self.labels_path[int(len(self.labels_path)*0.8): ]
            # self.labels_path = self.labels_path
    
    def __getitem__(self, index):
        label, mask = self.get_label_and_mask(index)
        file_a, file_b = self.get_files_data(index)
        if self.split in ('train', 'val'):
            if self.time_shuffle and np.random.binomial(1, 0.5):
                return file_a, file_b, label, mask
            else:
                return file_b, file_a, label, mask
        elif self.split in ('test_train', 'test_val'):
            label_path = self.labels_path[index]
            if self.time_shuffle and np.random.binomial(1, 0.5):
                return file_a, file_b, label, mask, label_path
            else:
                return file_b, file_a, label, mask, label_path
        # return {'path_a': file_a_path, 
        #         'path_b': file_b_path, 
        #         'path_label':label_path, 
        #         'file_a': file_a, 
        #         'file_b':file_b, 
        #         'label':label, 
        #         'mask': mask}
    

class SAR_CD_direct(SAR_CD_base):
    ''' directly extract the real and image part and push into model seperatedly '''
    def __init__(self, *args, data_format = 'save_space', **kwargs) -> None:
        super().__init__(*args, **kwargs)   
        self.data_format = data_format

        # 20% val, 80% train
        if self.split == 'train':
            self.labels_path = self.labels_path[0: int(len(self.labels_path)*0.8)]
        elif self.split == 'val':
            self.labels_path = self.labels_path[int(len(self.labels_path)*0.8): ]
            # self.labels_path = self.labels_path


    def __getitem__(self, index):
        ''' 先不管高分三号的数据集，先弄RADATSAT-2的数据 '''
        # generate label and its mask
        label_path = self.labels_path[index]
        label = lbm.read_change_label_png(label_path)-1
        mask = label<2      # 1 表示存在有效标记，0表示没有标记
        label[~mask] = 0    # 1 表示存在变化，0表示没有变化或没有标记数据
        
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

        file_path = []
        file_path.append(osp.join(label_dir.replace('label', 'data'), a_time, 'C3'))
        file_path.append(osp.join(label_dir.replace('label', 'data'), b_time, 'C3'))

        # get the file data
        slice_idx = re.search(r'-\d{4}-', label_path)
        if slice_idx is None:
            raise ValueError('can not find the wave code')
        slice_idx = int(slice_idx.group()[1:5])

        file = []
        for ii in range(2):
            # meta_info = psr.read_hdr(file_path[ii])
            # het = meta_info['lines']
            # wes = meta_info['samples']
            # patch_y, patch_x = lbm.get_corrds_from_slice_idx([het, wes], (512,512), slice_idx)
            file.append(psr.read_c3(osp.join(file_path[ii], str(slice_idx)), out=self.data_format)[[0, 5, 8]])
            
        # patch_y, patch_x = int(patch_y), int(patch_x)
        # file_a = file_a[:, patch_y: patch_y+512, patch_x:patch_x+512]
        # file_b = psr.read_c3(file_b_path, out=self.data_format)[:, patch_y: patch_y+512, patch_x:patch_x+512]
        # clip data
        for ii in range(2):
            file[ii][file[ii]>1] = 1
            file[ii][file[ii]<-1] = -1

        return torch.from_numpy(file[0]), torch.from_numpy(file[1]), torch.from_numpy(label).long(), torch.from_numpy(mask)

        # return torch.from_numpy(psr.rgb_by_c3(file[0])).permute(2,0,1), torch.from_numpy(psr.rgb_by_c3(file[1])).permute(2,0,1), torch.from_numpy(label).long(), torch.from_numpy(mask)

        # return {'path_a': file_path[0], 
        #         'path_b': file_path[1], 
        #         'path_label':label_path, 
        #         'file_a': file[0], 
        #         'file_b':file[1], 
        #         'label':label, 
        #         'mask': mask}


class SAR_CD_Hoekman(SAR_CD_base):
    def __init__(self, *args, time_shuffle=True, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.time_shuffle = time_shuffle
        # 20% val, 80% train
        if self.split == 'train':
            self.labels_path = self.labels_path[int(len(self.labels_path)*0.2):]
            # self.labels_path = self.labels_path[0: int(len(self.labels_path)*0.8)]
        elif self.split == 'val':
            self.labels_path = self.labels_path[:int(len(self.labels_path)*0.2)]
            # self.labels_path = self.labels_path[int(len(self.labels_path)*0.8): ]
            # self.labels_path = self.labels_path

    def __getitem__(self, index: int):
        label, mask = self.get_label_and_mask(index)

        # get the file path
        label_path = self.labels_path[index]
        label_dir = osp.split(label_path)[0]
        # two date time display format
        if osp.isfile(osp.join(label_dir, 'pin.txt')):   # mode 1
            re_exp = r'20\d{6}'
        elif osp.isfile(osp.join(label_dir, 'pin-2.txt')):   # mode 2
            re_exp = r'\d{2}[A-Z][a-z]{2}\d{4}'
        else:
            raise ValueError('unknown mode of label format')
        [a_time, b_time] = re.findall(re_exp, label_path)

        file_path = []
        file_path.append(osp.join(label_dir.replace('label', 'data'), a_time, 'C3'))
        file_path.append(osp.join(label_dir.replace('label', 'data'), b_time, 'C3'))

        # get the file data
        slice_idx = re.search(r'-\d{4}-', label_path)
        if slice_idx is None:
            raise ValueError('can not find the wave code')
        slice_idx = int(slice_idx.group()[1:5])
        file = []
        for ii in range(2):
            tmp = psr.read_c3(osp.join(file_path[ii], str(slice_idx)))
            tmp = psr.Hokeman_decomposition(tmp)
            # for jj in range(9):
            #     tmp[jj, :, :] = psr.min_max_contrast_median_map(10*np.log10(tmp[jj, :, :]))
            tmp = np.log10(tmp)
            file.append(tmp)
        
        if self.time_shuffle and np.random.binomial(1, 0.5):
            return torch.from_numpy(file[0]), torch.from_numpy(file[1]), label, mask
        else:
            return torch.from_numpy(file[1]), torch.from_numpy(file[0]), label, mask

        # return {'path_a': file_path[0], 
        #         'path_b': file_path[1], 
        #         'path_label':label_path, 
        #         'file_a': file[0], 
        #         'file_b':file[1], 
        #         'label':label, 
        #         'mask': mask}


class SAR_CD_tile_1(SAR_CD_base):
    ''' change detection using tiles randomly selected from a big picture with 512x512 pixels'''
    def __init__(self, *args, tile_size=32, mask_thres=3/4.0, label_thres=1/4.0, time_shuffle=False, **kargs):
        ''' tile_size should be the power of 2 '''
        super().__init__(*args, **kargs)
        self.mask_thres = mask_thres
        self.label_thres = label_thres
        if isinstance(tile_size, (list, tuple)):
            self.tile_size = tuple(tile_size)
        else:
            self.tile_size = (tile_size, tile_size)
        self.tiles_per_img = (512//self.tile_size[0])**2
        self.time_shuffle = time_shuffle
        print('two time phase shuffle:', time_shuffle)

        # 20% val, 80% train
        if self.split in ('train', 'test_train'):
            self.labels_path = self.labels_path[0: int(len(self.labels_path)*0.8)]
        elif self.split in ('val', 'test_val'):
            self.labels_path = self.labels_path[int(len(self.labels_path)*0.8): ]
            # self.labels_path = self.labels_path
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        if self.split in ['train', 'val']:
            return len(self.labels_path)*self.tiles_per_img
        elif self.split == ['test_train', 'test_val']:
            return len(self.labels_path)
        else:
            raise NotImplementedError
        
    def get_rand_slice_idx(self, mask):
        ''' get the slice index of a random selected tile, in which the valid pixels (with annotation) must greater than a threshold (0~1) '''
        # print('i am in')
        while True:
            tile_y = np.random.randint(0, 512-self.tile_size[0])
            tile_x = np.random.randint(0, 512-self.tile_size[1])
            tile_slice = [slice(tile_y, tile_y+self.tile_size[0]), slice(tile_x, tile_x+self.tile_size[1])]

            tile_mask = mask[tuple(tile_slice)]
            if tile_mask.sum()>=self.tile_size[0]*self.tile_size[1]*self.mask_thres:
                ''' only pick the tiles with enough annotationis'''
                # print('i am out')
                return tile_slice

    def __getitem__(self, index):
        if self.split in ['train', 'val']:
            img_idx, tile_idx = divmod(index, self.tiles_per_img)
            tile_coords = lbm.get_corrds_from_slice_idx((512, 512), self.tile_size, tile_idx)
            
            ''' label=1 表示存在变化，0表示没有变化 '''
            big_label, big_mask = self.get_label_and_mask(img_idx)
            tile_slice = [slice(tile_coords[0], tile_coords[0]+self.tile_size[0]), slice(tile_coords[1], tile_coords[1]+self.tile_size[1])]
            mask = big_mask[tuple(tile_slice)]
            # print('index:', index)
            # if index==4586:
            #     print('breakpoint')
            if mask.sum()<self.tile_size[0]*self.tile_size[1]*self.mask_thres:
                tile_slice = self.get_rand_slice_idx(big_mask)
                # print('random')
                # mask = big_mask[tuple(tile_slice)]

            
            label = big_label[tuple(tile_slice)].sum() >= self.tile_size[0]*self.tile_size[1]*self.label_thres
            # label.astype(np.int)
            files = self.get_files_data(img_idx)
            if self.data_type=='pauli':
                tile_slice.insert(0, slice(0, 3))
            elif self.data_type=='original':
                tile_slice.insert(0, slice(0, 9))
            else:
                raise NotImplementedError
            if self.time_shuffle and np.random.binomial(1, 0.5):
                return files[1][tuple(tile_slice)], files[0][tuple(tile_slice)], label, 
            else:
                return files[0][tuple(tile_slice)], files[1][tuple(tile_slice)], label,
            # return files[0][tuple(tile_slice)], files[1][tuple(tile_slice)], label, mask, big_label[tuple(tile_slice[1:])], img_idx, tile_idx,self.labels_path[img_idx], big_label,tile_slice
        elif self.split in ['test_train', 'test_val']:
            label_path = self.labels_path[index]
            label, mask = self.get_label_and_mask(index)
            files = self.get_files_data(index)
            return files[0], files[1], label, mask, label_path
        else:
            raise NotImplementedError

            

if __name__ == '__main__':
    print('pwd: ', os.getcwd())
    # print('sys path: ', sys.path)
    print('_'*20, '\n')
    
    opt = dict()
    opt['dataset_path'] = '/home/csl/data/SAR_CD/RS2'
    opt['data_format'] = 'save_space'
    opt['time_shuffle'] = True
    # opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # opt.add_argument('--dataset path', type=str, )
    
    
    ''' SAR_CD_tile_1 '''
    # dataset = SAR_CD_tile_1(root=opt['dataset_path'], split='train', tile_size=128)
    # print('size of dataset: ', len(dataset))
    # save_dir = './tmp/'
    # for ii in range(len(dataset)):
    #     file_a, file_b, label, mask, label_ori, img_idx, tile_idx, label_path, big_label, tile_slice = dataset.__getitem__(ii)
    #     print('label path:', label_path)
    #     print('image index:', img_idx, ', tile index', tile_idx)
    #     print('label:', label)
    #     print('tile slice:', tile_slice)
    #     print('-'*50)
    #     cv2.imwrite(save_dir+'fila_a.png', (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
    #     cv2.imwrite(save_dir+'fila_b.png', (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))
    #     cv2.imwrite(save_dir+'mask.png', (mask.numpy()*255).astype(np.uint8))
    #     cv2.imwrite(save_dir+'label.png', (label_ori.numpy()*255).astype(np.uint8))
    #     cv2.imwrite(save_dir+'big_label.png', (big_label.numpy()*255).astype(np.uint8))
    # # file_a = ff['file_a']
    # # file_b = ff['file_b']
    # # for ii in range(9):
    # #     plt.hist(file_a[ii, :, :])
    # #     plt.savefig(osp.join(save_dir, f'a-{ii}.png'))
    # #     plt.hist(file_b[ii, :, :])
    # #     plt.savefig(osp.join(save_dir, f'b-{ii}.png'))
    # #     # cv2.imwrite(osp.join(save_dir, f'a-{ii}.png'), 255*file_a[ii, :, :])
    # #     # cv2.imwrite(osp.join(save_dir, f'b-{ii}.png'), 255*file_b[ii, :, :])

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

    ''' SAR_CD_direct '''
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
    dataset = SAR_CD_intensities(root=opt['dataset_path'], split='val')
    print('size of dataset: ', len(dataset))
    save_dir = 'tmp/'
    for ii in range(len(dataset)):
        file_a, file_b, label, mask = dataset.__getitem__(ii)
        cv2.imwrite(save_dir+'file_a.png', (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
        cv2.imwrite(save_dir+'file_b.png', (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))
        cv2.imwrite(save_dir+'label.png', (label.numpy()*255).astype(np.uint8))
        cv2.imwrite(save_dir+'mask.png', (mask.numpy()*255).astype(np.uint8))
    # print('train set:')
    # for item in dataset.labels_path:
    #     print(item)
    # print('_'*100)
    # dataset = SAR_CD_intensities(root=opt['dataset_path'], split='test')
    # dataset.__getitem__(3)
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

    
