import os, sys
import argparse

from analyzer.data import Dataloader
from analyzer.data.data_raw import readvol, folder2Vol, savelabvol
from analyzer.data.data_vis import visvol

def create_arg_parser():
	'''
	Get arguments from command lines.
	'''
	parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
	parser.add_argument('--em', type=str, help='input directory em (path)')
	parser.add_argument('--gt', type=str, help='input directory gt (path)')

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

	dl = Dataloader(args.em, args.gt)
	em, gt = dl.load_chunk()
	visvol(em, gt)

	#em = args.em
	#gt = args.gt

	#h5data = folder2Vol(em)
	#data = savelabvol(h5data, 'human_em.h5', dataset='main')

	#print(data)


if __name__ == "__main__":
	main()
