import os, sys
import numpy as np
import time
import datetime
from collections import OrderedDict
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers

def build_monitor(cfg, output_path, mode='train'):
    '''building a tensorboard monitor for pytorch models.'''
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':', '-')
    log_dir = os.path.join(output_path, 'log' + '_' + date + '_' + time + '_' + mode)
    return Logger(cfg, log_dir)

class Logger(object):
    def __init__(self, cfg, log_dir=''):
        self.cfg = cfg
        self.log_dir = log_dir
        self.log_tb = SummaryWriter(self.log_dir)
        self.log_txt = open(os.path.join(self.log_dir, 'log.txt'), 'w')
        self.start_time = time.time()

        # Addding all the parameters to certain run.
        with open(os.path.join(self.log_dir, 'params.yaml'), 'w') as yfile:
            yaml.dump(dict(cfg), yfile, default_flow_style=False)

    def reset(self):
        pass

    def update(self, loss, iter, lr, epoch=0, acc=0.0, recon_loss=0.0, kld_loss=0.0):
        '''update the logger file.'''
        if self.log_tb is not None:
            self.log_tb.add_scalar('Loss', loss, iter)
            self.log_tb.add_scalar('Learning Rate', lr, iter)

        if self.log_txt is not None:
            if acc != 0.0:
                self.log_txt.write('accuracy %.4f \n' % (acc))
            elif recon_loss != 0.0 and kld_loss != 0.0:
                self.log_txt.write('[iteration %d] train_loss=%0.4f recon_loss=%0.4f kld_loss= %0.4f lr=%.5f\n epoch=%d \n' \
                % (iter, loss, recon_loss, kld_loss, lr, epoch))
            else:
                self.log_txt.write('[iteration %d] train_loss=%0.4f lr=%.5f\n epoch=%d \n' % (iter, loss, lr, epoch))
            self.log_txt.flush()

    def note_run_time(self, iter=0):
        '''calling this function will note the overall running time.'''
        with open(os.path.join(self.log_dir, 'run_time.txt'), 'w'):
            self.run_time.write('overall run_time: %.2f [iteration %d]' % (time.time() - self.start_time), iter)
