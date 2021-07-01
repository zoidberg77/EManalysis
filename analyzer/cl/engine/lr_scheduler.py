import torch
import numpy as np

class LRScheduler():
	'''General learning rule schedule object.'''
	def __init__(self,
				 optimizer: torch.optim.Optimizer,
				 base_lr: float,
				 num_epochs: int,
				 iter_per_epoch: int,
				 final_lr: float = 0.0,
				 warmup_epochs: int = 10,
				 warmup_lr: float = 0.0
				 ):
		self.optimizer = optimizer
		self.base_lr = base_lr
		self.final_lr = final_lr
		self.num_epochs = num_epochs
		self.iter_per_epoch = iter_per_epoch
		self.warmup_epochs = warmup_epochs
		self.warmup_lr = warmup_lr

		self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
		self.warmup_lr_schedule = np.linspace(self.warmup_lr, self.base_lr, self.warmup_iter)
		self.decay_iter = self.iter_per_epoch * (self.num_epochs - self.warmup_epochs)
		self.cosine_lr_schedule = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + np.cos(np.pi * np.arange(self.decay_iter) / self.decay_iter))

		self.lr_schedule = np.concatenate((self.warmup_lr_schedule, self.cosine_lr_schedule))
		self.iter = 0
		self.current_lr = 0

	def step(self):
		for param_group in self.optimizer.param_groups:
			lr = param_group['lr'] = self.lr_schedule[self.iter]

		self.iter += 1
		self.current_lr = lr
		return lr

	def get_lr(self):
		return self.current_lr
