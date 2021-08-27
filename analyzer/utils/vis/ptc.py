import h5py
from analyzer.data.data_vis import visptc

def vis_reconstructed_ptc(cfg, path=None):
    with h5py.File(cfg.PTC.MONITOR_PATH + cfg.PTC.RECONSTRUCTION_DATA, 'r') as ptcf:
        group = ptcf[list(ptcf.keys())[0]]
        for key in group.keys():
            obj = group[key]
            visptc(obj)

def vis_original_ptc(cfg, path=None):
    with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'r') as ptcf:
        group = ptcf[list(ptcf.keys())[0]]
        for key in group.keys():
            obj = group[key]
            visptc(obj)
