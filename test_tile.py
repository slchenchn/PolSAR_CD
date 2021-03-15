from operator import mod
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
cuda_idx = [2]
import argparse
import numpy as np
from numpy import ndarray
import random

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.metrics import runningScore, averageMeter

import yaml
import cv2 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch import nn
from torchvision import transforms

import mylib.polSAR_utils as psr
from mylib import labelme_utils as lbm

def test(args,cfg):
    ''' use the trained model to test '''

    # Setup random seeds
    torch.manual_seed(cfg.get('seed', 137))
    torch.cuda.manual_seed(cfg.get('seed', 137))
    np.random.seed(cfg.get('seed', 137))
    random.seed(cfg.get('seed', 137))

    # setup augmentations
    augs = cfg['train'].get('augmentations', None)
    data_aug = get_composed_augmentations(augs)

    # setup dataloader
    data_loader = get_loader(cfg['data']['dataloader'])
    data_path = cfg['data']['path']
    print('using dataset:', data_path, ', dataloader:', cfg['data']['dataloader'])
    if cfg['test']['train_set']:
        test_train_loader = data_loader(root=data_path, transfrom=None, split='test_train', augmentations=data_aug)
    if cfg['test']['val_set']:
        test_val_loader = data_loader(root=data_path, transform=None, split='test_val', augmentations=data_aug)
    
    # setup model
    model = get_model(cfg['model'])
    print('using model:', cfg['model']['arch'])
    device = f'cuda:{cuda_idx[0]}'
    model = model.to(device)
    # don't need run on multiple gpus
    # model = nn.DataParallel(model, device_ids=cuda_idx)

    # load model
    pth_path = cfg['test']['pth']
    if osp.isfile(pth_path):
        print('load model from checkpoint', pth_path)
        check_point = torch.load(pth_path)
        model.load_state_dict(check_point['model_state'])
    else:
        raise FileNotFoundError('can not find the specified .pth file')

    # setup metrics
    inc_metrics = runningScore()
    current_metrics = runningScore()

    # test
    tile_size = cfg['data']['tile_size']
    if not isinstance(tile_size, (tuple, list)):
        tile_size = (tile_size, tile_size)
    if cfg['test']['train_set']:
        test_loader = test_train_loader
    else:
        test_loader = test_val_loader
    tiles_per_image = test_loader.tiles_per_image
    model.eval()
    with torch.no_grad():
        for file_a, file_b, label, mask, label_path in test_loader:
            regid_pred = np.zeros_like(label)
            final_pred = regid_pred.zeros_like(label)

            # tile-wise change detection
            for tile_idx in range(tiles_per_image):
                tile_coords = lbm.get_corrds_from_slice_idx((512, 512), tile_size, tile_idx)
                tile_a = file_a[tile_coords[0]:tile_coords+tile_size[0], tile_coords[1]:tile_coords[1]+tile_size[1], :]
                tile_b = file_b[tile_coords[0]:tile_coords+tile_size[0], tile_coords[1]:tile_coords[1]+tile_size[1], :]
                tile = torch.cat((tile_a, tile_b), dim=0)
                tile_outputs = model(tile)
                tile_pred = tile_outputs.max(dim=0)[1]
                regid_pred[tile_coords[0]:tile_coords[0]+tile_size[0], tile_coords[1]:tile_coords[1]+tile_size[1]] = tile_pred

            # use file a to make superpixel segmentation
            segs = slic(file_a, n_segments=1024, compactness=0.5, min_size_factor=0.5, enforce_connectivity=True, convert2lab=False)
            for spix_idx in range(segs.max()+1):
                spix_region = segs==spix_idx
                final_pred[spix_region] = regid_pred[spix_region].sum()>spix_region.sum()/2
            
            # evaluate and save
            inc_metrics.update(label, final_pred, mask)
            current_metrics.update(label, final_pred, mask)
            score, cls_iou = current_metrics.get_scores()
            for k, v in cls_iou.items():
                print('{}: {}'.format(k, v))
            current_metrics.reset()
            save_path = label_path[:-4]+'_pred_.png'
            cv2.imwrite(save_path, final_pred)

        # ultimate evaluate
        score, cls_iou = inc_metrics.get_scores()
        for k, v in score.items():
            print('{}: {}'.format(k, v))
        for k, v in cls_iou.items():
            print('{}: {}'.format(k, v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="./runs/HRS_SAR_seg_4band/83014/EffUnet_HRS_SAR_seg_final_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="HRS_SAR_seg",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )
    # parser.add_argument(
    #     "--img_norm",
    #     dest="img_norm",
    #     action="store_true",
    #     default=True,
    #     help="Enable input image scales normalization [0, 1] \
    #                           | True by default",
    # )
    # parser.add_argument(
    #     "--img_path", nargs="?", type=str,
    #     default="./input_path/", help="Path of the input image"
    # )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default="./test_result/",
        help="Path of the output segmap",
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/tile.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(args,cfg)
