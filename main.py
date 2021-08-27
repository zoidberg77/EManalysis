import argparse
import sys
from glob import glob
import h5py
import torch
import pytorch_lightning as pl

from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader, PtcDataset, PairDataset
from analyzer.model.build_model import Clustermodel
from analyzer.utils.eval_model import Evaluationmodel
from analyzer.vae import train
from analyzer.vae.model.utils.pt import generate_volume_ptc, point_cloud
from analyzer.vae.model.random_ptc_ae import RandomPtcAe, RandomPtcDataModule
from analyzer.cl.trainer import CLTrainer
from analyzer.vae.model.vae import Vae, VaeDataModule

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
        dl.extract_scale_mitos_samples(parallel=False)
        return
    elif cfg.MODE.PROCESS == "train":
        print('--- Starting the training process for the vae --- \n')
        vae_model = Vae(cfg).double()
        vae_dataset = Dataloader(cfg)
        trainer = pl.Trainer(default_root_dir='datasets/vae/checkpoints', max_epochs=cfg.AUTOENCODER.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS)
        vae_datamodule = VaeDataModule(cfg=cfg, dataset=vae_dataset)
        trainer.fit(vae_model, vae_datamodule)
        return
    elif cfg.MODE.PROCESS == "infer":
        print('--- Starting the inference for the features of the vae. --- \n')
        return
    elif cfg.MODE.PROCESS == "ptcprep":
        dl = Dataloader(cfg)
        point_cloud(cfg, dl)
        return
    elif cfg.MODE.PROCESS == "ptctrain":
        print('--- Starting the training process for the vae based on point clouds. --- \n')
        ptcdl = PtcDataset(cfg)
        trainer = train.PtcTrainer(cfg=cfg, dataset=ptcdl, train_percentage=0.7, optimizer_type="adam")
        trainer.train()
        return
    elif cfg.MODE.PROCESS == "ptcinfer":
        print('--- Starting to infer the features of the autoencoder based on point clouds. --- \n')
        ptcdl = PtcDataset(cfg)
        trainer = train.PtcTrainer(cfg=cfg, dataset=ptcdl)
        trainer.save_latent_feature(m_version=cfg.PTC.MODEL_VERSION)
        return
    elif cfg.MODE.PROCESS == "rptctrain":
        print('--- Starting the training process for the vae based on point clouds(random). --- \n')
        rptc_model = RandomPtcAe(cfg).double()
        ptc_dataset = PtcDataset(cfg, sample_mode=cfg.PTC.SAMPLE_MODE,
                                 sample_size=cfg.PTC.RECON_NUM_POINTS)
        trainer = pl.Trainer(default_root_dir='datasets/vae/checkpoints', max_epochs=cfg.PTC.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS)
        ptc_datamodule = RandomPtcDataModule(cfg=cfg, dataset=ptc_dataset)
        trainer.fit(rptc_model, ptc_datamodule)
        return
    elif cfg.MODE.PROCESS == "rptcinfer":
        print('--- Starting the inference process for the vae based on point clouds(random). --- \n')
        last_checkpoint = glob('datasets/vae/checkpoints/lightning_logs/**/*.ckpt', recursive=True)[-1]
        rptc_model = RandomPtcAe(cfg).load_from_checkpoint(last_checkpoint, cfg=cfg).double()
        rptc_model.freeze()
        ptc_dataset = PtcDataset(cfg, sample_mode=cfg.PTC.SAMPLE_MODE,
                                 sample_size=cfg.PTC.RECON_NUM_POINTS)
        trainer = pl.Trainer(default_root_dir='datasets/vae/checkpoints', max_epochs=cfg.PTC.EPOCHS,
                             checkpoint_callback=False, gpus=cfg.SYSTEM.NUM_GPUS)
        ptc_datamodule = RandomPtcDataModule(cfg=cfg, dataset=ptc_dataset)

        with h5py.File(cfg.DATASET.ROOTF + 'ptc_shapef.h5', 'w') as f:
            f.create_dataset(name='id', shape=(len(ptc_dataset),))
            f.create_dataset(name='ptc_shape', shape=(len(ptc_dataset), 1024))
            f.create_group('ptc_reconstruction')
        trainer.test(model=rptc_model, test_dataloaders=ptc_datamodule.test_dataloader())
        return
    elif cfg.MODE.PROCESS == "cl":
        data = PairDataset(cfg)
        trainer = CLTrainer(cfg)
        trainer.train()
        return

    dl = Dataloader(cfg)
    model = Clustermodel(cfg, dl=dl)
    model.run()

if __name__ == "__main__":
    main()
