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
				 filters: List[int] =[64, 64, 64, 128, 1024],
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

		#net = tf.reshape(global_feat, [batch_size, -1])
		#end_points['embedding'] = net

		#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		#tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None,name=None)
		#net = tf_util.conv2d(input_image, 64, [1,point_dim], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)

		# --- decoding ---
		self.decoder = nn.Sequential(
					   nn.Linear(self.filters[4], 1024), nn.ReLU(),
					   nn.Linear(1024, 1024), nn.ReLU(),
					   nn.Linear(1024, (self.num_points * 3)),
					   )
		#self.decoder_input = nn.Sequential(nn.Linear(1024, np.prod(self.encoder_dim)), nn.LeakyReLU())
		#self.fc1 = nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU())
		#self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU())
		#self.fc3 = nn.Linear(1024, (self.num_points * 3))

		#net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
		#net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
		#net = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3')
		#net = tf.reshape(net, (batch_size, num_point, 3))


	def forward(self, x):
		#print(self.num_points)
		x = self.encoder(x)
		x = self.pool(x)
		x = torch.flatten(x, start_dim=1)
		x = self.decoder(x)
		x = x.view(x.size(0), x.size(0), -1, 3)
		return x
