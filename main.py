import argparse
import sys

import torch

from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader, PtcDataloader
from analyzer.data.random_ptc_dataset import RandomPtcDataset
from analyzer.model import Clustermodel
from analyzer.vae import train
from analyzer.vae.model.utils.pt import point_cloud
from analyzer.vae.model.random_ptc_vae import RandomPtcVae, RandomPtcDataModule

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

# RUN THE SCRIPT LIKE: $ python main.py --cfg configs/process.yaml
# Apply your specification within the .yaml file.

def create_arg_parser():
	'''
	Get arguments from command lines.
	'''
	parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
	parser.add_argument('--cfg', type=str, help='configuration file (path)')
	parser.add_argument('--mode', type=str, help='infer or train mode')

	return parser

def main():
	'''
	Main function.
	'''
	# input arguments are parsed.
	arg_parser = create_arg_parser()
	args = arg_parser.parse_args(sys.argv[1:])
	print("Command line arguments:")
	print(args)

	# configurations
	if args.cfg is not None:
		cfg = get_cfg_defaults()
		cfg.merge_from_file(args.cfg)
		if args.mode is not None:
			cfg.MODE.PROCESS = args.mode
		cfg.freeze()
		print("Configuration details:")
		print(cfg, '\n')
	else:
		cfg = get_cfg_defaults()
		cfg.freeze()
		print("Configuration details:")
		print(cfg, '\n')

	if cfg.MODE.PROCESS == "preprocessing":
		dl = Dataloader(cfg)
		em, labels, gt = dl.load_chunk()
		dl.extract_scale_mitos()
		return
	elif cfg.MODE.PROCESS == "ptcprep":
		dl = Dataloader(cfg)
		em, labels, gt = dl.load_chunk()
		_, gtfns = dl.get_fns()
		point_cloud(gtfns, cfg)
		return
	elif cfg.MODE.PROCESS == "ptctrain":
		print('--- Starting the training process for the vae based on point clouds. --- \n')
		ptcdl = PtcDataloader(cfg)
		trainer = train.PtcTrainer(cfg=cfg, dataset=ptcdl, train_percentage=0.7, optimizer_type="adam", loss_function="l1")
		trainer.train()
		return
	elif cfg.MODE.PROCESS == "ptcinfer":
		print('--- Starting to infer the features of the autoencoder based on point clouds. --- \n')
		ptcdl = PtcDataloader(cfg)
		trainer = train.PtcTrainer(cfg=cfg, dataset=ptcdl, train_percentage=0.7, optimizer_type="adam", loss_function="l1")
		trainer.save_latent_feature()
		return
	elif cfg.MODE.PROCESS == "train":
		for feature in cfg.AUTOENCODER.FEATURES:
			dl = Dataloader(cfg, feature=feature)
			print('--- Starting the training process of the {} autoencoder. --- \n'.format(feature))
			trainer = train.Trainer(dataset=dl, train_percentage=0.7, optimizer_type="adam", loss_function="l1", cfg=cfg)
			train_total_loss, test_total_loss = trainer.train()
			print("train loss: {}".format(train_total_loss))
			print("test loss: {}".format(test_total_loss))
		return
	elif cfg.MODE.PROCESS == "infer":
		print('--- Starting the inference for the features of the autoencoder. --- \n')
		for feature in cfg.AUTOENCODER.FEATURES:
			dl = Dataloader(cfg, feature=feature)
			trainer = train.Trainer(dataset=dl, train_percentage=0.7, optimizer_type="adam", loss_function="l1", cfg=cfg)
			trainer.save_latent_feature()
		return
	elif cfg.MODE.PROCESS == "rptctrain":
		print('--- Starting the training process for the vae based on point clouds(random). --- \n')
		rptc_model = RandomPtcVae(cfg).double()
		ptc_dataset = RandomPtcDataset(cfg)
		trainer = pl.Trainer(default_root_dir='datasets/vae/checkpoints')
		ptc_datamodule = RandomPtcDataModule(cfg=cfg, dataset=ptc_dataset)
		trainer.fit(rptc_model, ptc_datamodule)
		return

	dl = Dataloader(cfg)
	em, labels, gt = dl.load_chunk()

	from analyzer.data.data_vis import visvol
	visvol(em[0], labels[0], gt[0])

	#from analyzer.utils import Evaluationmodel
	#eval = Evaluationmodel(cfg, dl)
	#eval.create_gt_vector()

	#model = Clustermodel(cfg, em, gt, dl=dl)
	#model.visualize()
	#model.run()


if __name__ == "__main__":
	main()
