import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import *

class PTCvae(nn.Module):
    '''point cloud autoencoder. https://github.com/charlesq34/pointnet-autoencoder that directly consumes point
    point clouds, which well respects the permutation invariance of points in the input.

    @article{
        author = {Charles Ruizhongtai Qi and Hao Su and Kaichun Mo and Leonidas J. Guibas},
          title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
          journal   = {CoRR},
          volume    = {abs/1612.00593},
          year      = {2016},
          url       = {http://arxiv.org/abs/1612.00593}
    }
    '''
    def __init__(self,
                 num_points,
                 in_channel: int = 1,
                 out_channel: int = 1,
                 filters: List[int] = [64, 64, 64, 128, 512],
                 linear_layers: List[int] = [1024, 2048, 4096],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.filters = filters
        self.depth = len(self.filters)
        self.kernel_size = (3, 3)
        self.padding = (1, 1)

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        self.linear_layers = linear_layers
        self.num_points = num_points

        # --- encoding ---
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=self.filters[1])

        self.conv_in = nn.Sequential(
            conv2d_norm_act(self.in_channel, self.filters[0], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[0], self.filters[1], self.kernel_size, self.padding, **shared_kwargs)
        )
        self.conv_feat = nn.Sequential(
            conv2d_norm_act(self.filters[1], self.filters[2], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[2], self.filters[3], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[3], self.filters[4], self.kernel_size, self.padding, **shared_kwargs)
        )
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1,1))

        # --- decoding ---
        self.conv_decoder = nn.Sequential(
            trans_conv2d_norm_act(self.filters[4], 256, kernel_size=(2,2), stride=1, **shared_kwargs),
            trans_conv2d_norm_act(256, 256, kernel_size=(3,3), stride=1, **shared_kwargs),
            trans_conv2d_norm_act(256, 128, kernel_size=(4,4), stride=2, **shared_kwargs),
            trans_conv2d_norm_act(128, 64, kernel_size=(5,5), stride=3, **shared_kwargs),
            trans_conv2d_norm_act(64, 3, kernel_size=(1,1), stride=1, **shared_kwargs)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.filters[4], self.linear_layers[0]), nn.ReLU(),
            nn.Linear(self.linear_layers[0], self.linear_layers[1]), nn.ReLU(),
            nn.Linear(self.linear_layers[1], self.linear_layers[2]), nn.ReLU(),
            nn.Linear(self.linear_layers[2], (self.num_points * 3))
        )

    def forward(self, x):
        matrix3x3 = self.input_transform(torch.squeeze(x, 1).transpose(1, 2))
        x = torch.bmm(torch.squeeze(x, 1), matrix3x3)
        x = self.conv_in(torch.unsqueeze(x, 0).transpose(0, 1))

        # matrix64x64 = self.feature_transform(torch.squeeze(x, 0).transpose(1, 2))
        # x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1,2)
        x = self.conv_feat(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.decoder(x)
        x = x[:,None,:,None]
        # x = self.conv_decoder(x[(None,)*2].transpose(1, 3))
        x = x.view(x.size(0), x.size(1), -1, 3)
        return x

    def encoding(self, x):
        matrix3x3 = self.input_transform(torch.squeeze(x, 1).transpose(1, 2))
        x = torch.bmm(torch.squeeze(x, 1), matrix3x3)
        x = self.conv_in(torch.unsqueeze(x, 0).transpose(0, 1))

        #matrix64x64 = self.feature_transform(torch.squeeze(x, 0).transpose(1, 2))
        #x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1,2)
        x = self.conv_feat(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def decoding(self, x):
        x = self.decoder(x)
        x = x[:,None,:,None]
        # x = self.conv_decoder(x[(None,)*2].transpose(1, 3))
        x = x.view(x.size(0), x.size(1), -1, 3)
        return x


class PNAE(nn.Module):
    ''' Autoencoder using org. PointNet. '''
    def __init__(self,
                 latent: int = 100):
        super().__init__()
        self.encoder = PointNet()
        self.decoder = PNDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PNDecoder(nn.Module):
    ''' PointNet but the opposite.'''
    def __init__(self,
                 num_points: int = 4096,
                 filters: List[int] = [512, 1024, 4096],
                 latent: int = 100):
        super().__init__()
        self.num_points = num_points
        self.filters = filters
        self.latent = latent

        self.transform = Transformer()
        self.linear = nn.Sequential(
            nn.Linear(self.latent, self.filters[0]), nn.ReLU(),
            nn.Linear(self.filters[0], self.filters[1]), nn.ReLU(),
            nn.Linear(self.filters[1], self.filters[2]), nn.ReLU(),
            nn.Linear(self.filters[2], (self.num_points * 3))
        )

    def forward(self, x):
        x = self.linear(x)
        x = x[:,None,:,None]
        x = x.view(x.size(0), x.size(1), -1, 3)
        return x

class PointNet(nn.Module):
    ''' original PointNet after:
        @article{
            author = {Charles Ruizhongtai Qi and Hao Su and Kaichun Mo and Leonidas J. Guibas},
              title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
              journal   = {CoRR},
              volume    = {abs/1612.00593},
              year      = {2016},
              url       = {http://arxiv.org/abs/1612.00593}
        }
    '''
    def __init__(self,
                 filters: List[int] = [1024, 512, 256],
                 latent: int = 100):
        super().__init__()
        self.filters = filters
        self.latent = latent

        self.transform = Transformer()
        self.fc1 = nn.Linear(self.filters[0], self.filters[1])
        self.fc2 = nn.Linear(self.filters[1], self.filters[2])
        self.fc3 = nn.Linear(self.filters[2], self.latent)


        self.bn1 = nn.BatchNorm1d(self.filters[1])
        self.bn2 = nn.BatchNorm1d(self.filters[2])
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x, matrix3x3, matrix64x64 = self.transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

class Transformer(nn.Module):
    '''uses TNets to perform transformation.'''
    def __init__(self,
                 filters: List[int] = [64, 128, 1024],
                 kernel_size: int = 1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.k = [3, 64]
        self.input_transform = TNet(k = self.k[0])
        self.feature_transform = TNet(k = self.k[1])

        # Convolutional Layer
        self.conv_1 = conv1d_norm_act(self.k[0], self.filters[0], self.kernel_size)
        self.conv_2 = nn.Sequential(
            conv1d_norm_act(self.filters[0], self.filters[1], self.kernel_size),
            conv1d_norm_act(self.filters[1], self.filters[2], self.kernel_size)
        )

    def forward(self, x):
        matrix3x3 = self.input_transform(torch.squeeze(x, 1).transpose(1, 2))
        x = torch.bmm(torch.squeeze(x, 1), matrix3x3)
        x = self.conv_1(x.transpose(1, 2))

        matrix64x64 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)

        x = self.conv_2(x)
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        return x, matrix3x3, matrix64x64


class TNet(nn.Module):
    '''Transformer Network that predicts an affine transformation matrix and directly apply this
    transformation to the coordinates of input points.

    Args:
        k: dimension of matrix transformation.
        filters: filter planes.
    '''
    def __init__(self,
                 k: int = 3,
                 filters: List[int] = [64, 128, 1024],
                 linear_layers: List[int] = [512, 256],
                 kernel_size: int = 1):
        super().__init__()
        self.k = k
        self.filters = filters
        self.linear_layers = linear_layers
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            conv1d_norm_act(self.k, self.filters[0], self.kernel_size),
            conv1d_norm_act(self.filters[0], self.filters[1], self.kernel_size),
            conv1d_norm_act(self.filters[1], self.filters[2], self.kernel_size)
        )

        self.fc1 = nn.Linear(self.filters[2], self.linear_layers[0])
        self.fc2 = nn.Linear(self.linear_layers[0], self.linear_layers[1])
        self.fc3 = nn.Linear(self.linear_layers[1], self.k * self.k)

        #self.bn4 = nn.BatchNorm1d(self.linear_layers[0])
        #self.bn5 = nn.BatchNorm1d(self.linear_layers[1])

    def forward(self, x):
        x = self.conv(x)
        pool = nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        #x = self.relu(self.bn4(self.fc1(flat)))
        #x = self.relu(self.bn5(self.fc2(x)))
        x = self.relu(self.fc1(flat))
        x = self.relu(self.fc2(x))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(x.size(0), 1, 1)
        if x.is_cuda:
            init = init.cuda()
        matrix = self.fc3(x).view(-1, self.k, self.k) + init
        return matrix
