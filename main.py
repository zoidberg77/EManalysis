import argparse
import sys
from glob import glob

import numpy as np
import h5py
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from analyzer.cl.trainer import CLTrainer
from analyzer.config import get_cfg_defaults
from analyzer.data import Dataloader, PtcDataset
from analyzer.model.build_model import Clustermodel
from analyzer.vae import train
from analyzer.vae.model.random_ptc_ae import RandomPtcAe, RandomPtcDataModule
from analyzer.vae.model.utils.pt import point_cloud
from analyzer.vae.model.vae import Vae, VaeDataModule
from analyzer.vae.model.ptc_ae import FoldingNet, PtcAeDataModule

# RUN THE SCRIPT LIKE: $ python main.py --cfg configs/process.yaml
# Apply your specification within the .yaml file.

def create_arg_parser():
    '''Get arguments from command lines.'''
    parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
    parser.add_argument('--cfg', type=str, help='configuration file (path)')
    parser.add_argument('--mode', type=str, help='infer or train mode')

    return parser

def main():
    '''Main function.
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
        print(dl.prep_data_info())
        exit()
        dl.extract_scale_mitos_samples()
        return
    elif cfg.MODE.PROCESS == "train":
        print('--- Starting the training process for the vae --- \n')
        vae_model = Vae(cfg)
        vae_dataset = Dataloader(cfg)
        trainer = pl.Trainer(default_root_dir=cfg.AUTOENCODER.MONITOR_PATH + 'checkpoints', max_epochs=cfg.AUTOENCODER.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS, gradient_clip_val=0.5, stochastic_weight_avg=True)
        vae_datamodule = VaeDataModule(cfg=cfg, dataset=vae_dataset)
        trainer.fit(vae_model, vae_datamodule)
        trainer.save_checkpoint(cfg.AUTOENCODER.MONITOR_PATH + "vae.ckpt")
        vae_model.save_logging()
        return
    elif cfg.MODE.PROCESS == "infer":
        print('--- Starting the inference for the features of the vae. --- \n')
        with h5py.File(cfg.DATASET.ROOTD + "mito_samples.h5", "a") as mainf:
            size_needed = len(mainf["id"])
            if "output" not in mainf:
                mainf.create_dataset("output", mainf["chunk"].shape)

            with h5py.File(cfg.DATASET.ROOTF+'shapef.h5', 'w') as h5f:
                h5f.create_dataset("id", (size_needed, ))
                h5f.create_dataset("shape", (size_needed, cfg.AUTOENCODER.LATENT_SPACE))
                h5f.create_dataset("output", mainf["chunk"].shape)


        vae_model = Vae(cfg)
        vae_model.load_from_checkpoint(checkpoint_path=cfg.AUTOENCODER.MONITOR_PATH + "vae.ckpt", cfg=cfg)
        vae_dataset = Dataloader(cfg)
        trainer = pl.Trainer(default_root_dir=cfg.AUTOENCODER.MONITOR_PATH + 'checkpoints', max_epochs=cfg.AUTOENCODER.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS)
        vae_datamodule = VaeDataModule(cfg=cfg, dataset=vae_dataset)
        trainer.test(vae_model, vae_datamodule.test_dataloader())

        with h5py.File(cfg.DATASET.ROOTD + "mito_samples.h5", "a") as mainf:
            with h5py.File(cfg.DATASET.ROOTF+'shapef.h5', 'a') as shapef:
                for i, e in enumerate(shapef["output"]):
                    mainf["output"][i] = e
                del shapef["output"]
                samples = {}
                for i, e in enumerate(shapef["id"]):
                    samples[e] = shapef["shape"][i]
                dl = Dataloader(cfg)
                for e in dl.prep_data_info():
                    k = e["id"]
                    if k not in samples.keys():
                        samples[k] = np.zeros((cfg.AUTOENCODER.LATENT_SPACE,))

                del shapef["shape"]
                del shapef["id"]
                shapef.create_dataset("id", (len(samples.keys()), ))
                shapef.create_dataset("shape", (len(samples.keys()), cfg.AUTOENCODER.LATENT_SPACE))

                c = 0
                for k,v in sorted(samples.items()):
                    shapef["shape"][c] = v
                    shapef["id"][c] = k
                    c += 1
        return
    elif cfg.MODE.PROCESS == "ptcprep":
        dl = Dataloader(cfg)
        point_cloud(cfg, dl)
        return
    elif cfg.MODE.PROCESS == "ptctrain":
        print('--- Starting the training process for the vae based on point clouds. --- \n')
        ptcdl = PtcDataset(cfg)
        ptc_module = FoldingNet(cfg)
        trainer = pl.Trainer(default_root_dir=cfg.PTC.MONITOR_PATH + 'checkpoints', max_epochs=cfg.PTC.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS)
        ptc_datamodule = PtcAeDataModule(cfg=cfg, dataset=ptcdl)
        trainer.fit(ptc_module, datamodule=ptc_datamodule)
        trainer.save_checkpoint(cfg.PTC.MONITOR_PATH + "ptc_ae.ckpt")
        return
    elif cfg.MODE.PROCESS == "ptcinfer":
        print('--- Starting to infer the features of the autoencoder based on point clouds. --- \n')
        size_needed = 0
        with h5py.File(cfg.DATASET.ROOTD + "pts.h5", "r") as mainf:
            size_needed = len(mainf["labels"])

        with h5py.File(cfg.DATASET.ROOTF+'shapef.h5', 'w') as h5f:
            h5f.create_dataset("id", (size_needed, ))
            h5f.create_dataset("shape", (size_needed, cfg.PTC.LATENT_SPACE))
            h5f.create_dataset("output", (size_needed, cfg.PTC.SAMPLE_SIZE, 3))

        ptcdl = PtcDataset(cfg)
        ptc_module = FoldingNet(cfg)
        ptc_module.load_from_checkpoint(checkpoint_path=cfg.PTC.MONITOR_PATH + "ptc_ae.ckpt", cfg=cfg)
        trainer = pl.Trainer(default_root_dir=cfg.PTC.MONITOR_PATH + 'checkpoints', max_epochs=cfg.PTC.EPOCHS,
                             gpus=cfg.SYSTEM.NUM_GPUS)
        ptc_datamodule = PtcAeDataModule(cfg=cfg, dataset=ptcdl)
        trainer.test(ptc_module, datamodule=ptc_datamodule)

        return
    elif cfg.MODE.PROCESS == "cltrain":
        print('--- Starting the training process for the contrastive learning setup. --- \n')
        trainer = CLTrainer(cfg)
        trainer.train()
        return
    elif cfg.MODE.PROCESS == "cltest":
        print('--- Starting the testing process for the contrastive learning setup. --- \n')
        trainer = CLTrainer(cfg)
        trainer.test()
    elif cfg.MODE.PROCESS == "clinfer":
        print('--- Extracting the features using the Contrastive Learning model. --- \n')
        trainer = CLTrainer(cfg)
        trainer.infer_feat_vector()
    else:
        dl = Dataloader(cfg)
        model = Clustermodel(cfg, dl=dl)
        model.run()

if __name__ == "__main__":
    main()
