import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import *

class PTCvae(nn.Module):
    '''point cloud autoencoder. https://github.com/charlesq34/pointnet-autoencoder

    @InProceedings{Yang_2018_CVPR,
        author = {Yang, Yaoqing and Feng, Chen and Shen, Yiru and Tian, Dong},
        title = {FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
    }
    '''

    def __init__(self,
                 batch_size,
                 filters=[64, 64, 64, 128, 1024]):
        pass
        self.filters = filters
        self.batch_size = batch_size

        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        shared_kwargs =
        {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }
        #num_point = point_cloud.get_shape()[1].value
        #point_dim = point_cloud.get_shape()[2].value

        # --- encoding ---
        #for in range(len(self.filters)):
            #self.conv_1 = conv2d_norm_act(in_channel, self.filters[0], self.kernel_size, self.padding, **shared_kwargs)

        self.conv_1 = conv2d_norm_act(in_channel, self.filters[0], self.kernel_size, self.padding, **shared_kwargs)
        self.conv_2 = conv2d_norm_act(in_channel, self.filters[1], self.kernel_size, self.padding, **shared_kwargs)
        self.conv_3 = conv2d_norm_act(in_channel, self.filters[2], self.kernel_size, self.padding, **shared_kwargs)
        self.conv_4 = conv2d_norm_act(in_channel, self.filters[3], self.kernel_size, self.padding, **shared_kwargs)
        self.conv_5 = conv2d_norm_act(in_channel, self.filters[4], self.kernel_size, self.padding, **shared_kwargs)

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None,name=None)
        #net = tf_util.conv2d(input_image, 64, [1,point_dim], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
        # --- decoding ---


    def forward(self, x):
        pass
