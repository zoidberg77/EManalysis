import os, sys
import h5py
from analyzer.data.data_vis import visptc

def vis_reconstructed_ptc(cfg, path=None):
    with h5py.File(os.path.join(path, cfg.PTC.RECONSTRUCTION_DATA), 'r') as ptcf:
        group = ptcf[list(ptcf.keys())[0]]
        for key in group.keys():
            obj = group[key]
            visptc(obj)

def vis_original_ptc(cfg, path=None):
    with h5py.File(path, 'r') as ptcf:
        group = ptcf.get('ptcs')
        for key in group.keys():
            obj = group[key]
            visptc(obj)

def vis_sampled_ptc(cfg, path=None):
    with h5py.File(path, 'r') as ptcf:
        for _, key in enumerate(list(ptcf.keys())):
            obj = ptcf[key]
            visptc(obj)
