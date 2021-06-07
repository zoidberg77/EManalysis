import h5py

from analyzer.data.data_vis import visptc


def vis_reconstructed_ptc(cfg):
    with h5py.File(cfg.AUTOENCODER.OUTPUT_FOLDER + "ptc_shapef.h5", 'r') as ptcf:
        group = ptcf['ptc_reconstruction']
        for obj in group:
            visptc(obj)


def vis_original_ptc(cfg):
    with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'r') as ptcf:
        group = ptcf['ptcs']
        for key in group.keys():
            obj = group[key]
            visptc(obj)
