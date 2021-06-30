import torch
import numpy as np
from tqdm import tqdm
from analyzer.data.augmentation.augmentor import Augmentor
from analyzer.cl.model import get_model
from analyzer.data import PairDataset
from analyzer.cl.engine.loss import similarity_func

class CLTrainer():
	'''
		Trainer object, enabling constrastive learning framework.
		:params cfg: (yacs.config.CfgNode): YACS configuration options.
	'''
	def __init__(self, cfg):
		self.cfg = cfg
		self.num_epochs = 1
		self.device = 'cpu'
		if cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
			self.device = 'cuda'
		self.model = get_model(self.cfg).to(self.device)

		self.dataset = PairDataset(self.cfg)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=cfg.SSL.BATCH_SIZE,
		shuffle=False, pin_memory=True)

	def train(self):
		for epoch in tqdm(range(self.num_epochs)):
			self.model.train()

			for idx, (x1, x2) in enumerate(self.dataloader):

				print(x1.shape)

				self.model.zero_grad()
				z1, p1, z2, p2 = self.model.forward(x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True))
				loss = similarity_func(p1, z2) / 2 + similarity_func(p2, z1) / 2
				print(loss)
				loss = data_dict['loss'].mean()
				# loss.backward()
				# optimizer.step()
				# lr_scheduler.step()
				# data_dict.update({'lr':lr_scheduler.get_lr()})
				return

	def save_checkpoint():
		pass
