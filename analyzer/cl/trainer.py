import os, sys
import torch
import numpy as np
from tqdm import tqdm
from analyzer.data.augmentation.augmentor import Augmentor
from analyzer.cl.model import get_model
from analyzer.data import PairDataset
from analyzer.cl.engine.loss import similarity_func
from analyzer.cl.engine.optimizer import build_optimizer, build_lr_scheduler

class CLTrainer():
	'''
		Trainer object, enabling constrastive learning framework.
		:params cfg: (yacs.config.CfgNode): YACS configuration options.
	'''
	def __init__(self, cfg):
		self.cfg = cfg
		self.epochs = self.cfg.SSL.EPOCHS
		self.output_dir = self.cfg.SSL.OUTPUT_MODEL_PATH
		self.device = 'cpu'
		if self.cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
			self.device = 'cuda'
		self.model = get_model(self.cfg).to(self.device)

		self.dataset = PairDataset(self.cfg)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.SSL.BATCH_SIZE,
		shuffle=False, pin_memory=True)
		self.optimizer = build_optimizer(self.cfg, self.model)
		self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer, len(self.dataloader))

	def train(self):
		for epoch in range(0, self.epochs):
			self.model.train()
			for idx, ((x1, x2), labels) in enumerate(self.dataloader):

				self.model.zero_grad()
				z1, p1, z2, p2 = self.model.forward(x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True))
				loss = similarity_func(p1, z2) / 2 + similarity_func(p2, z1) / 2
				loss = loss.mean()
				loss.backward()
				self.optimizer.step()
				self.lr_scheduler.step()

			self.save_checkpoint(epoch)

	def save_checkpoint(self, idx: int):
		'''Save the model at certain checkpoints.'''
		# state = {'iteration': idx + 1,
		# 		 'state_dict': self.model.state_dict(),
		# 		 'optimizer': self.optimizer.state_dict(),
		# 		 'lr_scheduler': self.lr_scheduler.state_dict()}

		state = self.model.state_dict()
		filename = 'checkpoint_%05d.pth.tar' % (idx + 1)
		filename = os.path.join(self.output_dir, filename)
		torch.save(state, filename)
