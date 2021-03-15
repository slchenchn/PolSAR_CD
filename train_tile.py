import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
cuda_idx = [1]
import os.path as osp
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
import glob
import natsort
import re
import matplotlib.pyplot as plt

from mylib import types
import args
import utils

def train(cfg, writer, logger, run_id):
    
    # Setup random seeds
    # torch.manual_seed(cfg.get('seed', 137))
    # torch.cuda.manual_seed(cfg.get('seed', 137))
    # np.random.seed(cfg.get('seed', 137))
    # random.seed(cfg.get('seed', 137))

    torch.backends.cudnn.benchmark = True

    # Setup Augmentations
    augmentations = cfg['train'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataloader'])
    data_path = cfg['data']['path']

    logger.info("Using dataset: {}".format(data_path))

    tile_size = cfg['data']['tile_size']
    t_loader = data_loader(
        data_path,
        transform=None,
        split=cfg['data']['train_split'],
        tile_size=tile_size,
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        transform=None,
        tile_size=tile_size,
        split=cfg['data']['val_split'],
        )
    logger.info(f'num of train samples: {len(t_loader)} \nnum of val samples: {len(v_loader)}')

    train_data_len = len(t_loader)
    batch_size = cfg['train']['batch_size']
    epoch = cfg['train']['train_epoch']
    train_iter = int(np.ceil(train_data_len / batch_size) * epoch)
    logger.info(f'total train iter: {train_iter}')
    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['train']['batch_size'], 
                                  num_workers=cfg['train']['n_workers'], 
                                  shuffle=True,
                                  persistent_workers=True,
                                  drop_last=True)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['train']['batch_size'], 
                                num_workers=cfg['train']['n_workers'],
                                persistent_workers=True,
                                  )

    # Setup Model
    model = get_model(cfg['model'], n_classes)
    # print('model:\n', model)
    logger.info("Using Model: {}".format(cfg['model']['arch']))
    device = f'cuda:{cuda_idx[0]}'
    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=cuda_idx)      #自动多卡运行，这个好用

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['train']['optimizer'].items() 
                        if k != 'name'}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    scheduler = get_scheduler(optimizer, cfg['train']['lr_schedule'])
    loss_fn = get_loss_function(cfg)
    # logger.info("Using loss {}".format(loss_fn))

    # set checkpoints
    start_iter = 0
    if cfg['train']['resume'] is not None:
        if os.path.isfile(cfg['train']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['train']['resume'])
            )
            checkpoint = torch.load(cfg['train']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['train']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['train']['resume']))

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)
    val_loss_meter = averageMeter()
    train_time_meter = averageMeter()
    time_meter_val=averageMeter()

    flag = True

    val_rlt_f1=[]
    val_rlt_OA=[]
    best_fwIoU_now = 0
    best_fwIoU_iter_till_now = 0

    # train
    it = start_iter
    num_positive = 0
    model.train()   
    while it <= train_iter and flag:
        for (file_a, file_b, label) in trainloader:

            # caculate distribution of samples
            num_positive += label.sum()

            it += 1
            # print('iteration', it)
            start_ts = time.time()    
            file = torch.cat((file_a, file_b), dim=1).to(device)
            # file_a = file_a.to(device)            
            # file_b = file_b.to(device)            
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(file)

            loss = loss_fn(input=outputs, target=label)
            loss.backward()
            # print('conv11: ', model.conv11.weight.grad, model.conv11.weight.grad.shape)
            # print('conv21: ', model.conv21.weight.grad, model.conv21.weight.grad.shape)
            # print('conv31: ', model.conv31.weight.grad, model.conv31.weight.grad.shape)

            # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`
            optimizer.step()
            scheduler.step()
            
            train_time_meter.update(time.time() - start_ts)
            time_meter_val.update(time.time() - start_ts)

            if (it + 1) % cfg['train']['print_interval'] == 0:
                fmt_str = "train:\nIter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(it + 1,
                                           train_iter,
                                           loss.item(),      #extracts the loss’s value as a Python float.
                                           train_time_meter.avg / cfg['train']['batch_size'])
                train_time_meter.reset()
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), it+1)
                writer.add_scalar('num_positve', num_positive/(cfg['train']['batch_size']*cfg['train']['print_interval']), it+1)
                num_positive = 0

            if (it + 1) % cfg['train']['val_interval'] == 0 or \
               (it + 1) == train_iter:
                model.eval()            # change behavior like drop out
                with torch.no_grad():   # disable autograd, save memory usage
                    for (file_a_val, file_b_val, label_val) in valloader:  
                        file_val = torch.cat((file_a_val, file_b_val), dim=1).to(device)
                        # file_a_val = file_a_val.to(device)            
                        # file_b_val = file_b_val.to(device)

                        outputs = model(file_val)
                        label_val = label_val.to(device)
                        val_loss = loss_fn(input=outputs, target=label_val)
                        val_loss_meter.update(val_loss.item())

                        # tensor.max with return the maximum value and its indices
                        label_val = label_val.cpu().numpy()
                        pred = outputs.max(dim=1)[1].cpu().numpy()
                        running_metrics_val.update(label_val, pred)

                # lr_now = optimizer.param_groups[0]['lr']
                # logger.info(f'lr: {lr_now}')
                # writer.add_scalar('lr', lr_now, it+1)
                writer.add_scalars('loss/val_loss', {'train':loss.item(),'val':val_loss_meter.avg}, it+1)
                logger.info("Iter %d, val Loss: %.4f" % (it + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()

                # for k, v in score.items():
                #     logger.info('{}: {}'.format(k, v))
                #     writer.add_scalar('val_metrics/{}'.format(k), v, it+1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, it+1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                avg_f1 = score["Mean_F1"]
                OA=score["Overall_Acc"]
                fw_IoU = score["FreqW_IoU"]
                # val_rlt_f1.append(avg_f1)
                # val_rlt_OA.append(OA)

                if fw_IoU >= best_fwIoU_now and it>200:
                    best_fwIoU_now = fw_IoU
                    correspond_meanIou = score["Mean_IoU"]
                    best_fwIoU_iter_till_now = it+1

                    state = {
                        "epoch": it + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_fwIoU": best_fwIoU_now,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(), "{}_{}_best_model.pkl".format(cfg['model']['arch'],cfg['data']['dataloader']))
                    torch.save(state, save_path)

                    logger.info("best_fwIoU_now =  %.8f" % (best_fwIoU_now))
                    logger.info("Best fwIoU Iter till now= %d" % (best_fwIoU_iter_till_now))

                iter_time=time_meter_val.avg
                time_meter_val.reset()
                remain_time = iter_time * (train_iter - it)
                m, s = divmod(remain_time, 60)
                h, m = divmod(m, 60)
                if s != 0:
                    train_time = "Remain train time = %d hours %d minutes %d seconds \n" % (h, m, s)
                else:
                    train_time = "Remain train time : train completed.\n"
                logger.info(train_time)
                model.train()  
 
            if (it + 1) == train_iter:
                flag = False
                logger.info("Use the Sar_seg_band3,val_interval: 30")
                break
            
    logger.info("best_fwIoU_now =  %.8f" % (best_fwIoU_now))
    logger.info("Best fwIoU Iter till now= %d" % (best_fwIoU_iter_till_now))

    state = {"epoch": it + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_fwIoU": best_fwIoU_now,
    }
    save_path = os.path.join(writer.file_writer.get_logdir(), "{}_{}_last_model.pkl".format(cfg['model']['arch'],cfg['data']['dataloader']))
    torch.save(state, save_path)

if __name__ == "__main__":
    cfg = args.get_argparser(r'configs/tile.yml')
    del cfg.test
    torch.backends.cudnn.benchmark = True
    

    # generate work dir
    run_id = utils.get_work_dir(osp.join(r'runs', cfg.model.arch + '_' + cfg.train.loss.name + '_' + cfg.train.optimizer.name + '_' + cfg.train.lr.name))
    writer = SummaryWriter(log_dir=run_id)
    config_fig = types.dict2fig(cfg.to_flatten_dict())
    # plt.savefig(r'./tmp/ff.png')
    writer.add_figure('config', config_fig, close=True)
    # writer.add_hparams(types.flatten_dict_summarWriter(cfg), {'a': 'b'})
    writer.flush()
    # logger
    logger = get_logger(run_id)

    # print('RUNDIR: {}'.format(logdir))
    logger.info(f'RUNDIR: {run_id}')
    shutil.copy(cfg.config_file, run_id)


    train(cfg, writer, logger)
    logger.info(f'RUNDIR:{run_id}')
