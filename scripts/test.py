import argparse
import sys, os
import h5py
import numpy as np

# adding the right path.
parent = os.path.abspath(os.getcwd())
sys.path.append(parent)

from analyzer.config import get_cfg_defaults
from analyzer.utils.vis.ptc import *

def create_arg_parser():
    '''Get arguments from command lines.'''
    parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
    parser.add_argument('--cfg', type=str, help='configuration file (path)')
    parser.add_argument('--mode', type=str, help='infer or train mode')

    return parser

def main():
    '''testing function.'''
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    print("Command line arguments:")
    print(args)

    ### configurations
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

    ### section free to use.
    vis_reconstructed_ptc(cfg, path='models/ptc/human/run_2021-08-31/')
    #vis_sampled_ptc(cfg, 'datasets/rat/mc_pts.h5')

    # from analyzer.data import Dataloader, PtcDataset, PairDataset
    # ptcdl = PtcDataset(cfg)
    # print(ptcdl[10682])
    # from analyzer.data.data_vis import visptc
    # with h5py.File('datasets/human/pts.h5', 'r') as ptcf:
    #     group = ptcf.get('ptcs')
    #     obj = np.array(group[str(17920)])
    #     print(obj.shape)
    #     visptc(obj)

if __name__ == "__main__":
    main()
