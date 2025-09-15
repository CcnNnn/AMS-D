# Copyright (c) 2021-present LG CNS Corp.
# original code from:  
# "Self-Knowledge Distillation with Progressive Refinement of Targets"
# Kyungyul Kim, ByeongMoon Ji, Doyoung Yoon, Sangheum Hwang, ICCV 2021
# GitHub: https://github.com/lgcnsai/PS-KD-Pytorch

# =====================================================================
# further modified by Nuo Chen (2025)
# added "init_seed" function

'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
'''

import torch
from torch import nn
import torch.distributed as dist

import os, logging
import numpy as np
import random


def init_seed(seed=6013):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    torch.use_deterministic_algorithms(True, warn_only=True)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

        
def paser_config_save(args,PATH):
    import json
    with open(PATH+'/'+'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

        
def set_logging_defaults(logdir, args):
    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])
    # log cmdline argumetns
    logger = logging.getLogger('main')
    if is_main_process():
        logger.info(args)
        
        
def check_args(args):
    # --epoch
    assert args.end_epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args


def save_data_pandas(data_dict, save_path):
    import pandas as pd
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if  os.path.exists(save_path):
        parent_data= pd.read_csv(save_path)
        child_data = pd.DataFrame(data_dict, index=[0])
        data = parent_data._append(child_data)
        data.to_csv(save_path, index=False, mode='w')
    else:
        data = pd.DataFrame(data_dict, index=[0])
        data.to_csv(save_path, index=False)
    return data


def save_data_jpg(data_dict, data_type=['acc'], plot_title='Acc', xylabel={'x':'Epoch','y':'Acc'}, save_path='Train_curve.jpg'):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.cla()
    x = range(1, len(data_dict[list(data_dict.keys())[0]])+1)
    for type in data_type:
        plt.plot(x, data_dict[type], label=type)
    plt.title(plot_title)
    plt.legend()
    plt.xlabel(xylabel['x'])
    plt.ylabel(xylabel['y'])
    plt.savefig(save_path)