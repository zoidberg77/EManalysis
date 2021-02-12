import argparse
import sys

from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader
from analyzer.model import Clustermodel
from analyzer.model import FeatureExtractor
from analyzer.vae.dataset import MitoDataset


# RUN THE SCRIPT LIKE: $ python main.py --em datasets/human/human_em_export_8nm/ --gt datasets/human/human_gt_export_8nm/ --cfg configs/process.yaml

def create_arg_parser():
    '''
    Get arguments from command lines.
    '''
    parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
    parser.add_argument('--em', type=str, help='input directory em (path)')
    parser.add_argument('--gt', type=str, help='input directory gt (path)')
    parser.add_argument('--cfg', type=str, help='configuration file (path)')
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

    # configurations
    if args.cfg is not None:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.cfg)
        cfg.freeze()
        print("Configuration details:")
        print(cfg)

    if args.mode == "autoencoder":
        dataset = MitoDataset(em_path=cfg.DATASET.EM_PATH, gt_path=cfg.DATASET.LABEL_PATH,
                              mito_volume_file_name=cfg.AUTOENCODER.OUPUT_FILE_VOLUMES,
                              mito_volume_dataset_name=cfg.AUTOENCODER.DATASET_NAME,
                              target_size=cfg.AUTOENCODER.TARGET, lower_limit=cfg.AUTOENCODER.LOWER_BOUND,
                              upper_limit=cfg.AUTOENCODER.UPPER_BOUND, chunks_per_cpu=cfg.AUTOENCODER.CHUNKS_CPU,
                              ff=cfg.DATASET.FILE_FORMAT,
                              region_limit=cfg.AUTOENCODER.REGION_LIMIT, cpus=cfg.SYSTEM.NUM_CPUS)
        dataset.extract_scale_mitos()
        exit()

    dl = Dataloader(cfg, chunk_size=(2, 4096, 4096))
    em, gt = dl.load_chunk(vol='both')

    #dl.precluster(mchn='cluster')

    #fex = FeatureExtractor(em, gt, args.em, args.gt, dprc='iter')
    #tmp = fex.compute_seg_dist()
    # print(tmp)
    # fex.save_feat_dict(tmp, 'sizef.json')

    model = Clustermodel(em, gt, dl=dl, alg='kmeans', clstby='bysize')
    model.run()


if __name__ == "__main__":
    main()
