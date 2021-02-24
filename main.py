import argparse
import sys

from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader
from analyzer.model import Clustermodel
from analyzer.model import FeatureExtractor
from analyzer.vae import train

import matplotlib.pyplot as plt
import numpy as np


# RUN THE SCRIPT LIKE: $ python main.py --em datasets/human/human_em_export_8nm/ --gt datasets/human/human_gt_export_8nm/ --cfg configs/process.yaml

def create_arg_parser():
    '''
    Get arguments from command lines.
    '''
    parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
    parser.add_argument('--em', type=str, help='input directory em (path)')
    parser.add_argument('--gt', type=str, help='input directory gt (path)')
    parser.add_argument('--cfg', type=str, help='configuration file (path)')

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
        cfg.freeze()
        print("Configuration details:")
        print(cfg)
    else:
        cfg = get_cfg_defaults()
        cfg.freeze()
        print("Configuration details:")
        print(cfg)

    dl = Dataloader(cfg)
    em, gt = dl.load_chunk(vol='both')


    if cfg.MODE.PROCESS == "infer":
        #dataset.extract_scale_mitos()
        dl.extract_scale_mitos()
        return
    elif cfg.MODE.PROCESS == "train":
        print('--- Starting the training process of the autoencoder. --- \n')
        trainer = train.Trainer(dataset=dl, train_percentage=0.7, optimizer_type="adam", loss_function="l1", cfg=cfg)
        train_total_loss, test_total_loss = trainer.train()
        print("train loss: {}".format(train_total_loss))
        print("test loss: {}".format(test_total_loss))

        return

    # dl.precluster(mchn='cluster')

    fex = FeatureExtractor(cfg, em, gt)
    #tmp = fex.compute_seg_circ()
    #print(tmp)
    #fex.save_feat_h5(tmp, 'circf')

    model = Clustermodel(cfg, em, gt, dl=dl, clstby='bysize')
    model.load_features()
    #model.run()


if __name__ == "__main__":
    main()
