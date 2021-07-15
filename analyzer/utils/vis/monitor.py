import os, sys
import numpy as np
import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def build_monitor():
    '''building a tensorboard monitor for pytorch models.'''
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':', '-')
    log_path = os.path.join(cfg.DATASET.OUTPUT_PATH, 'log'+ date + '_' + time)


class Logger(object):
    def __init__(self, log_dir=''):
        self.log_tb = SummaryWriter(log_dir)
        self.log_txt = open(os.path.join(log_dir, 'log.txt'), 'w')
