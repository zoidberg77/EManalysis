from analyzer.data.augmentation.augmentor import Augmentor
from analyzer.cl.model import get_model

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
		model = get_model(self.cfg).to(self.device)

	def train(self):
		for epoch in global_progress:
			model.train()

			for idx, ((images1, images2), labels) in enumerate():

				model.zero_grad()
				data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
				loss = data_dict['loss'].mean()
				loss.backward()
				optimizer.step()
				lr_scheduler.step()
				data_dict.update({'lr':lr_scheduler.get_lr()})

	def save_checkpoint():
		pass
