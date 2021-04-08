import argparse
import sys

from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader, PtcDataloader
from analyzer.model import Clustermodel
from analyzer.vae import train
from analyzer.vae.model.utils.pt import point_cloud
from analyzer.data.data_vis import visptc

import matplotlib.pyplot as plt
import numpy as np

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

	dl = Dataloader(cfg)
	em, gt = dl.load_chunk(vol='both')

	if cfg.MODE.PROCESS == "preprocessing":
		dl.extract_scale_mitos()
		return
	elif cfg.MODE.PROCESS == "ptcprep":
		_, gtfns = dl.get_fns()
		point_cloud(gtfns, cfg)
		return
	elif cfg.MODE.PROCESS == "ptctrain":
		print('--- Starting the training process for the vae based on point clouds. --- \n')
		ptcdl = PtcDataloader(cfg)
		num_pts = 10000
		trainer = train.PtcTrainer(cfg=cfg, dataset=ptcdl, num_points=num_pts, train_percentage=0.7, optimizer_type="adam", loss_function="l1")
		trainer.train()
		#ptc = np.squeeze(ptcdl[1325], axis=0)
		#visptc(np.squeeze(ptc, axis=0))
		#import open3d as o3d
		#point_cloud = o3d.geometry.PointCloud()
		#open3d.geometry.VoxelGrid()
		#point_cloud.points = o3d.utility.Vector3dVector(ptc)
		#voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.005)
		#bbox = point_cloud.get_axis_aligned_bounding_box()
		#print(np.array(bbox))
		#o3d.visualization.draw_geometries([voxel_grid])
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

	model = Clustermodel(cfg, em, gt, dl=dl)
	#model.visualize()
	model.run()


if __name__ == "__main__":
	main()
