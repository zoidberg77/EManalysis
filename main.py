import os, sys
import argparse

from analyzer.data import Dataloader
from analyzer.model import Clustermodel
from analyzer.model import FeatureExtractor

# RUN THE SCRIPT LIKE: $ python main.py --em datasets/human/human_em_export_8nm/ --gt datasets/human/human_gt_export_8nm/

def create_arg_parser():
	'''
	Get arguments from command lines.
	'''
	parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
	parser.add_argument('--em', type=str, help='input directory em (path)')
	parser.add_argument('--gt', type=str, help='input directory gt (path)')
	parser.add_argument('--mode', type=str, help='cluster or autoencoder mode', default='cluster')

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

	dl = Dataloader(args.em, args.gt, chunk_size=(2,4096,4096), mito_slice_limit=40)
	em, gt = dl.load_chunk(vol='both')

	fex = FeatureExtractor(em, gt)
	fex.compute_seg_size()

	if args.mode == "autoencoder":
		dl.extract_scale_mitos(chunk_size=12)
		exit()

	model = Clustermodel(em, gt, dl=dl, alg='kmeans', clstby='bydist')
	model.run()


if __name__ == "__main__":
	main()
