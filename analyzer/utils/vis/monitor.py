import os, sys
import numpy as np
import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def build_monitor(cfg, output_path):
    '''building a tensorboard monitor for pytorch models.'''
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':', '-')
    log_dir = os.path.join(output_path, 'log' + date + '_' + time)
    return Logger(log_dir)

class Logger(object):
    def __init__(self, log_dir=''):
        self.log_tb = SummaryWriter(log_dir)
        self.log_txt = open(os.path.join(log_dir, 'log.txt'), 'w')

    def reset(self):
        pass

    def update(self, loss, iter, lr):
        if self.log_tb is not None:
            self.log_tb.add_scalar('Loss', loss, iter)
            self.log_tb.add_scalar('Learning Rate', lr, iter)

            #self.log_tb.add_figure('Loss', loss, iter)
            #plt.close('all')

        if self.log_txt is not None:
            self.log_txt.write('[iteration %d] train_loss=%0.4f lr=%.5f\n' % (iter, loss, lr))
            self.log_txt.flush()
