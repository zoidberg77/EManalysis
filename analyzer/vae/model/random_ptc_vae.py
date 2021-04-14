from typing import List

import numpy as np
import torch
from chamferdist import ChamferDistance
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision import transforms

from analyzer.vae.model.block import conv2d_norm_act
from torch.utils.data import random_split, DataLoader

class RandomPtcVae(pl.LightningModule):
    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 1,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 latent_space: int = 100,
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'None',
                 pooling: bool = False,
                 sample_size=1000,
                 lr = 1e-4,
                 **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filters = filters
        self.depth = len(self.filters)
        self.kernel_size = (1, 1)
        self.padding = (1, 1)
        self.latent_space = latent_space
        self.sample_size = sample_size
        self.lr = lr
        self.dist = ChamferDistance()

        shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        encoder_modules = []
        encoder_modules.append(nn.Conv2d(self.in_channel, self.filters[0], kernel_size=self.kernel_size, padding=self.padding))
        encoder_modules.append(nn.ReLU())
        for i, filter in enumerate(self.filters[:-1]):
            encoder_modules.append(nn.Conv2d(filter, self.filters[i+1], kernel_size=self.kernel_size, padding=self.padding))
            encoder_modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_modules)
        self.encoder_dim = self.encoder(torch.zeros(1, 1, self.sample_size, 3)).size()[1:]

        self.encoder_last = nn.Linear(np.prod(self.encoder_dim),latent_space)
        self.fc = nn.Linear(latent_space, np.prod(self.encoder_dim))
        self.decoder_modules = []
        for i in range(len(self.filters)-1, 0, -1):
            self.decoder_modules.append(
                nn.ConvTranspose2d(self.filters[i], self.filters[i-1], kernel_size=self.kernel_size, padding=self.padding))
            self.decoder_modules.append(nn.ReLU())
            #self.decoder_modules.append(nn.BatchNorm2d(self.filters[i-1]))

        self.decoder_modules.append(
            nn.ConvTranspose2d(self.filters[0], self.out_channel, kernel_size=self.kernel_size, padding=self.padding))

        self.decoder_modules.append(nn.ReLU())

        self.decoder = nn.Sequential(*self.decoder_modules)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        feats = self.encoder_last(x)
        z = self.fc(feats)
        z = z.view(-1, *self.encoder_dim)
        x_hat = self.decoder(z)
        return x_hat


    def step(self, batch, batch_idx):
        x = batch
        #print(x.min().item(), x.max().item(), x.shape)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        feats = self.encoder_last(x)
        z = self.fc(feats)
        z = z.view(-1, *self.encoder_dim)
        x_hat = self.decoder(z)
        #print(x_hat.min().item(), x_hat.max().item())
        loss = self.loss(x_hat, batch)
        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch.double(), batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, reconstruction, org_data):
        rec = torch.squeeze(reconstruction, dim=1)
        org = torch.squeeze(org_data, dim=1)
        #loss = [self.dist(rec[batch].float(), org[batch].float()) for batch in range(rec.shape[0])]
        loss = self.dist(rec.float(), org.float())
        #torch.mean(torch.Tensor(loss))
        return loss





class RandomPtcDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset = dataset

    def setup(self, stage=None):

        train_length = int(0.7 * len(self.dataset))
        test_length = len(self.dataset) - train_length
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, (train_length, test_length))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        pass