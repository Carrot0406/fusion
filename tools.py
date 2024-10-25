# 计算准确率代码，保存结果文件夹代码
import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import ipdb
from sklearn.metrics import average_precision_score


def para_name(opt):
    if opt.modality == 'v':
        name_para = 'net_v_only={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(

            opt.net_v,
            opt.method,
            opt.batch_size,
            opt.lr_decay,
            opt.lr,
            opt.lrd_rate
        )
    if opt.modality == 'c':
        name_para = 'net_c_only={}~method={}~lr={}'.format(

            opt.net_s,
            opt.method,
            opt.lr,
            
        )
    if opt.modality == 'v+c':
        name_para = '~net_v={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(

            opt.net_v,
            opt.method,
            opt.batch_size,
            opt.lr_decay,
            opt.lr,
            opt.lrd_rate
        )
    return name_para


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)