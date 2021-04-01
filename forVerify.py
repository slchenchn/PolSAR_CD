'''
Author: Shuailin Chen
Created Date: 2021-03-11
Last Modified: 2021-03-12
	content: 
'''
'''
Author: Shuailin Chen
Created Date: 2021-03-11
Last Modified: 2021-03-11
	content: 
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# cuda_idx = [0]
import os.path as osp
import sys
from torchvision.transforms.functional import to_tensor
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import glob
import natsort
import re
import logging
from mylib import nestargs
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms
from torch.nn import Module
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.loader.PolSAR_CD2 import *
from mylib import types
from mylib import file_utils as fu
import args
import utils


# if __name__=='__main__':
#     save_dir = './tmp'
#     ds = PolSAR_CD_base(root=r'/data/csl/SAR_CD/GF3', split='train', data_format='pauli', 
#     # augments=Compose([RandomHorizontalFlip(0.5)])
#     # augments=Compose([RandomVerticalFlip(0.5)])
#     augments=Compose([RandomRotation(180)])
#     )
#     idx = 70
#     ds.__getitem__(idx)

#     file_a, file_b = ds.get_files_data(idx)
#     label, mask = ds.get_label_and_mask(idx)
#     ds.statistics()
#     cv2.imwrite(osp.join(save_dir, 'fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
#     cv2.imwrite(osp.join(save_dir, 'fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
#     cv2.imwrite(osp.join(save_dir, 'mask.png'), (mask.numpy()*255).astype(np.uint8))
#     cv2.imwrite(osp.join(save_dir, 'label.png'), (label.numpy()*255).astype(np.uint8))
#     print('done')