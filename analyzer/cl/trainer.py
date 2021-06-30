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
		self.num_epochs = 1
		self.device = 'cpu'
		if cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
			self.device = 'cuda'
		self.model = get_model(self.cfg).to(self.device)

		self.dataset = PairDataset(self.cfg)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=cfg.SSL.BATCH_SIZE,
		shuffle=False, pin_memory=True)
		self.optimizer = build_optimizer(self.cfg, self.model)
		self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)

	def train(self):
		for epoch in tqdm(range(self.num_epochs)):
			self.model.train()

			for idx, (x1, x2) in enumerate(self.dataloader):

				self.model.zero_grad()
				z1, p1, z2, p2 = self.model.forward(x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True))
				loss = similarity_func(p1, z2) / 2 + similarity_func(p2, z1) / 2
				loss = loss.mean()
				loss.backward()
				self.optimizer.step()
				self.lr_scheduler.step()
				return

	def save_checkpoint():
		pass
