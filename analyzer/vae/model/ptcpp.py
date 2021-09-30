import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from .block import *
from .ptc_vae import TNet
from analyzer.cl.model.resnet import ResNet3D, ResNet3DMM

class PTCPP(nn.Module):
    ''' point cloud autoencoder. Based on:
    @article{
        author = {Remelli, Edoardo and Baque, Pierre and Fua, Pascal},
          title     = {NeuralSampler: Euclidean Point Cloud Auto-Encoder and Sampler},
          year      = {2019},
          url       = {http://arxiv.org/pdf/1901.09394v1}
    }
    '''
    def __init__(self,
                 num_points: int = 5000,
                 in_channel: int = 1,
                 pn_filters: List[int] = [64, 64, 64, 128, 512],
                 linear_layers: List[int] = [4096, 8192, 10000],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **kwargs):
        super(PTCPP, self).__init__()
        self.num_points = num_points
        self.in_channel = in_channel
        self.pn_filters = pn_filters
        self.linear_layers = linear_layers
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # --- encoding PointNet ---
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=self.pn_filters[1])

        self.conv_in = nn.Sequential(
            conv2d_norm_act(self.in_channel, self.pn_filters[0], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.pn_filters[0], self.pn_filters[1], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.pn_filters[1], self.pn_filters[2], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.pn_filters[2], self.pn_filters[3], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.pn_filters[3], self.pn_filters[4], self.kernel_size, self.padding, **shared_kwargs)
        )

        self.pool = nn.AdaptiveMaxPool3d(output_size=(64, 64, 64))

        self.resnet = ResNet3D()

        # --- decoding ---
        self.resnetmm = ResNet3DMM()
        self.linear_decoder = nn.Sequential(
            nn.Linear(3500, self.linear_layers[0]), nn.ReLU(),
            nn.Linear(self.linear_layers[0], self.linear_layers[1]), nn.ReLU(),
            nn.Linear(self.linear_layers[1], self.linear_layers[2]), nn.ReLU(),
            nn.Linear(self.linear_layers[2], (self.num_points * 3))
        )

    def forward(self, x):
        x = self.encoding(x)
        x = self.decoding(x)
        return x

    def encoding(self, x):
        x = self.conv_in(x)
        x = self.pool(x)
        x = self.resnet(torch.unsqueeze(x, 0))
        x = torch.flatten(x, start_dim=1)
        return x

    def decoding(self, x):
        x = x[:, :, None, None, None]
        x = self.resnetmm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_decoder(x)
        x = x[:,None,:,None]
        x = x.view(x.size(0), x.size(1), -1, 3)
        return x
