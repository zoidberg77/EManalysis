import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet3D(nn.Module):
    '''point cloud autoencoder. https://github.com/charlesq34/pointnet-autoencoder

    @InProceedings{Yang_2018_CVPR,
        author = {Yang, Yaoqing and Feng, Chen and Shen, Yiru and Tian, Dong},
        title = {FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
    } 
    '''

    def __init__(self):
        pass
