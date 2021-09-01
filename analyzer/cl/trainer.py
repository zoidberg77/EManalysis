import os, sys
import torch
import numpy as np
from tqdm import tqdm
from analyzer.data.augmentation.augmentor import Augmentor
from analyzer.cl.model import get_model
from analyzer.data import PairDataset
from analyzer.cl.engine.loss import similarity_func
from analyzer.cl.engine.optimizer import build_optimizer, build_lr_scheduler
from analyzer.utils.vis.monitor import build_monitor

class CLTrainer():
	'''
	Trainer object, enabling constrastive learning framework.
	:params cfg: (yacs.config.CfgNode): YACS configuration options.
	'''
	def __init__(self, cfg):
		self.cfg = cfg
		self.epochs = self.cfg.SSL.EPOCHS
		self.device = 'cpu'
		if self.cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
			self.device = 'cuda'
		self.model = get_model(self.cfg).to(self.device)

		# Setting up the dataset.
		self.dataset = PairDataset(self.cfg)
		train_length = int(self.cfg.SSL.TRAIN_PORTION * len(self.dataset))
		train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, (train_length, len(self.dataset) - train_length))
		self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.PTC.BATCH_SIZE, shuffle=False)
		self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.PTC.BATCH_SIZE, shuffle=False)

		# Setting up the optimizer, lr & logger.
		self.optimizer = build_optimizer(self.cfg, self.model)
		self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer, len(self.train_dl))
		self.logger = build_monitor(self.cfg)

		# Setting the outputpath for each run.
		if self.cfg.MODE.PROCESS == 'cltrain':
			time_now = str(datetime.datetime.now()).split(' ')
			if os.path.exists(os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])):
				self.output_path = os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])
			else:
				self.output_path = os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])
				os.mkdir(self.output_path)
		elif self.cfg.MODE.PROCESS == 'clinfer':
			self.state_model = self.cfg.SSL.STATE_MODEL
			self.output_path = self.cfg.SSL.MODEL.rsplit('/', 1)[0]
		else:
			raise ValueError('No valid process. Choose \'cltrain\' or \'clinfer\'.')

	def train(self):
		counter = 0
		running_loss = list()
		for epoch in range(0, self.epochs):
			self.model.train()
			for idx, ((x1, x2), labels) in enumerate(self.train_dl):

				self.model.zero_grad()
				z1, p1, z2, p2 = self.model.forward(x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True))
				loss = similarity_func(p1, z2) / 2 + similarity_func(p2, z1) / 2
				loss = loss.mean()
				loss.backward()
				running_loss.append(loss.item())
				self.optimizer.step()
				self.lr_scheduler.step()

				if not i % self.cfg.SSL.LOG_INTERVAL:
					self.logger.update((sum(running_loss) / len(running_loss)), counter, self.lr_scheduler.get_lr(), epoch)
				counter = counter + 1

				break

			self.save_checkpoint(epoch)

	def test(self):
		pass

	def save_checkpoint(self, idx: int):
		'''Save the model at certain checkpoints.'''
		# state = {'iteration': idx + 1,
		# 		 'state_dict': self.model.state_dict(),
		# 		 'optimizer': self.optimizer.state_dict(),
		# 		 'lr_scheduler': self.lr_scheduler.state_dict()}

		state = self.model.state_dict()
		#filename = 'checkpoint_%05d.pth.tar' % (idx + 1)
		filename = 'cl_model_{}.pt'.format(idx)
		filename = os.path.join(self.output_path, filename)
		torch.save(state, filename)
