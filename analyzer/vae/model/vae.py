import json
from typing import List

import h5py
import numpy as np
import torch
from chamferdist import ChamferDistance
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision import transforms

from analyzer.vae.model.block import conv2d_norm_act, conv3d_norm_act, BasicBlock3d, BasicBlock3dSE
from torch.utils.data import random_split, DataLoader

from analyzer.vae.model.utils import model_init


class Vae(pl.LightningModule):
    block_dict = {
        'residual': BasicBlock3d,
        'residual_se': BasicBlock3dSE,
    }

    def __init__(self,
                 cfg,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 1,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 input_shape=(64, 64, 64),
                 lr=1e-3,
                 **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.filters = filters
        self.depth = len(self.filters)
        self.kernel_size = (1, 1)
        self.padding = 0
        self.cfg = cfg
        self.stride = 1
        self.lr = lr
        super().__init__()
        assert len(filters) == len(isotropy)
        self.depth = len(filters)
        self.latent_space = self.cfg.AUTOENCODER.LATENT_SPACE
        self.input_shape = self.cfg.AUTOENCODER.TARGET
        if is_isotropic:
            isotropy = [True] * self.depth

        self.pooling = pooling
        block = self.block_dict[block_type]

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernel_size(is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
                                       padding=padding_io, **shared_kwargs)
        self.conv_out = conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='bn')

        # encoding path
        self.down_layers = nn.ModuleList()
        self.out_layer = torch.sigmoid

        for i in range(self.depth):
            kernel_size, padding = self._get_kernel_size(isotropy[i])
            previous = max(0, i - 1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **shared_kwargs),
                block(filters[i], filters[i], **shared_kwargs))
            self.down_layers.append(layer)

        self.encoder_dim = self.get_dim_after_encoder()

        self.mu = nn.Sequential(
            nn.Linear(np.prod(self.encoder_dim), self.latent_space),
            nn.LeakyReLU()
        )
        self.log_var = nn.Sequential(
            nn.Linear(np.prod(self.encoder_dim), self.latent_space),
            nn.LeakyReLU()
        )
        self.decoder_input = nn.Sequential(nn.Linear(self.latent_space, np.prod(self.encoder_dim)), nn.LeakyReLU())

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernel_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j - 1], kernel_size,
                                padding=padding, **shared_kwargs),
                block(filters[j - 1], filters[j - 1], **shared_kwargs)])
            self.up_layers.append(layer)

        self.logging_array = []
        self.inference = False
        # initialization
        model_init(self)

    def forward(self, x):
        x = self.conv_in(x)
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x

        x = self.down_layers[-1](x)
        x = torch.flatten(x, start_dim=1)
        latent_space = None
        if self.inference:
            latent_space = x

        log_var = self.log_var(x)
        mu = self.mu(x)
        x = self.reparameterize(mu, log_var)
        x = self.decoder_input(x)

        x = x.view(-1, *self.encoder_dim)

        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)

        x = self.conv_out(x)
        x = self.out_layer(x)
        return x, mu, log_var, latent_space

    def step(self, batch, batch_idx):
        reconstruction, mu, log_var, latent_space = self.forward(batch)
        loss, recon_loss, kld_loss = self.loss(reconstruction, batch, mu, log_var)
        self.logging_array.append({"recon_loss": recon_loss.item(), "kld_loss": kld_loss.item(), "loss": loss.item()})
        return loss, {"recon_loss": recon_loss, "kld_loss": kld_loss, "loss": loss}, reconstruction, latent_space

    def training_step(self, batch, batch_idx):
        raw_x, y = batch
        loss, logs, reconstruction, _ = self.step(raw_x, batch_idx)

        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_x, y = batch
        loss, logs, reconstruction, _ = self.step(raw_x, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, reconstruction, input, mu, log_var):

        recons_loss = torch.nn.functional.l1_loss(reconstruction, input, reduction="mean")
        kld_weight = 1
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss /= self.cfg.AUTOENCODER.LATENT_SPACE
        kld_loss *= kld_weight
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss

    def test_step(self, batch, batch_idx):
        self.inference = True
        raw_x, y = batch
        loss, logs, reconstruction, latent_space = self.step(raw_x, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)

        with h5py.File(self.cfg.DATASET.ROOTD + "mito_samples.h5", "a") as mainf:
            mainf["output"][y] = reconstruction
            obj_id = mainf["id"][y]
        with h5py.File(self.cfg.DATASET.ROOTF + "shapef.h5", "a") as featuref:
            featuref["id"][y] = obj_id
            featuref["shape"][y] = latent_space

        return loss

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `False`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x + y

    def _get_kernel_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1

        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return (1, 2, 2)
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)

        return nn.Identity()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_dim_after_encoder(self):
        out = self.conv_in(torch.zeros(1, 1, *self.input_shape))

        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            out = self.down_layers[i](out)
            down_x[i] = out

        out = self.down_layers[-1](out)
        return out.size()[1:]

    def save_logging(self):
        with open(self.cfg.DATASET.ROOTD+"log.json", 'w') as fp:
            json.dump(self.logging_array, fp)


class VaeDataModule(pl.LightningDataModule):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.cpus = cfg.SYSTEM.NUM_CPUS
        self.batch_size = cfg.AUTOENCODER.BATCH_SIZE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.double())
            # transforms.Lambda(self.helper_pickle)
        ])
        self.dataset = dataset

    def setup(self, stage=None):
        train_length = int(0.7 * len(self.dataset))
        test_length = len(self.dataset) - train_length
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, (train_length, test_length))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.cpus, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.cpus, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=self.cpus, shuffle=False)
