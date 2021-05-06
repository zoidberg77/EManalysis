from typing import List

import h5py
import numpy as np
import torch
from chamferdist import ChamferDistance
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision import transforms

from analyzer.vae.model.block import conv2d_norm_act
from torch.utils.data import random_split, DataLoader


class RandomPtcAe(pl.LightningModule):
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
                 cfg,
                 in_channel: int = 1,
                 out_channel: int = 1,
                 filters: List[int] = [64, 64, 64, 128, 512],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 lr=1e-4,
                 **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.filters = filters
        self.depth = len(self.filters)
        self.kernel_size = (1, 1)
        self.padding = (1, 1)
        self.dist = ChamferDistance()
        self.lr = lr
        self.cfg = cfg

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        self.linear = 1024
        # self.latent_space = latent_space
        self.num_points = cfg.AUTOENCODER.PTC_NUM_POINTS

        self.encoder = nn.Sequential(
            conv2d_norm_act(self.in_channel, self.filters[0], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[0], self.filters[1], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[1], self.filters[2], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[2], self.filters[3], self.kernel_size, self.padding, **shared_kwargs),
            conv2d_norm_act(self.filters[3], self.filters[4], self.kernel_size, self.padding, **shared_kwargs)
        )
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # --- decoding ---
        self.decoder = nn.Sequential(
            nn.Linear(self.filters[4], self.linear), nn.ReLU(),
            nn.Linear(self.linear, self.linear), nn.ReLU(),
            nn.Linear(self.linear, (self.num_points * 3)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(0), -1, 3)
        return x

    def step(self, batch, batch_idx):
        x_hat = self.forward(batch)
        loss = self.loss(batch, x_hat)
        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        raw_x, y = batch
        loss, logs = self.step(raw_x, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_x, y = batch
        loss, logs = self.step(raw_x, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, reconstruction, org_data):
        rec = torch.squeeze(reconstruction, dim=1)
        org = torch.squeeze(org_data, dim=1)
        # loss = [self.dist(rec[batch].float(), org[batch].float()) for batch in range(rec.shape[0])]
        loss = self.dist(rec.float(), org.float())
        # torch.mean(torch.Tensor(loss))
        return loss

    def test_step(self, batch, batch_idx):
        raw_x, y = batch
        x = self.encoder(raw_x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        with h5py.File('features/ptc_shapef.h5', 'a') as f:
            f['id'][batch_idx] = y
            latent_space = x[0]
            f['ptc_shape'][batch_idx] = latent_space.cpu()
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(0), -1, 3)
        loss = self.loss(raw_x, x)
        return loss


class RandomPtcDataModule(pl.LightningDataModule):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.cpus = cfg.SYSTEM.NUM_CPUS
        self.batch_size = cfg.AUTOENCODER.BATCH_SIZE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.double())
        ])
        self.dataset = dataset

    def setup(self, stage=None):
        train_length = int(0.7 * len(self.dataset))
        test_length = len(self.dataset) - train_length
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, (train_length, test_length))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.cpus, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.cpus, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=self.cpus, shuffle=False)
