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
				 num_points,
				 in_channel: int = 1,
				 out_channel: int = 1,
				 filters: List[int] =[64, 64, 64, 128, 512],
				 pad_mode: str = 'replicate',
				 act_mode: str = 'elu',
				 norm_mode: str = 'bn',
				 **kwargs):
		super().__init__()
		self.in_channel = in_channel
		self.filters = filters
		self.depth = len(self.filters)
		self.kernel_size = (1, 1)
		self.padding = (1, 1)

		shared_kwargs = {
			'pad_mode': pad_mode,
			'act_mode': act_mode,
			'norm_mode': norm_mode}

		self.linear = 1024
		#self.latent_space = latent_space
		self.num_points = num_points

		self.encoder = nn.Sequential(
			conv2d_norm_act(self.in_channel, self.filters[0], self.kernel_size, self.padding, **shared_kwargs),
			conv2d_norm_act(self.filters[0], self.filters[1], self.kernel_size, self.padding, **shared_kwargs),
			conv2d_norm_act(self.filters[1], self.filters[2], self.kernel_size, self.padding, **shared_kwargs),
			conv2d_norm_act(self.filters[2], self.filters[3], self.kernel_size, self.padding, **shared_kwargs),
			conv2d_norm_act(self.filters[3], self.filters[4], self.kernel_size, self.padding, **shared_kwargs)
		)
		self.pool = nn.AdaptiveMaxPool2d(output_size=(1,1))

		# --- decoding ---
		self.decoder = nn.Sequential(
					   nn.Linear(self.filters[4], self.linear), nn.ReLU(),
					   nn.Linear(self.linear, self.linear), nn.ReLU(),
					   nn.Linear(self.linear, (self.num_points * 3)),
					   )

	def forward(self, x):
		#print(self.num_points)
		x = self.encoder(x)
		x = self.pool(x)
		x = torch.flatten(x, start_dim=1)
		x = self.decoder(x)
		x = x.view(x.size(0), x.size(0), -1, 3)
		return x
