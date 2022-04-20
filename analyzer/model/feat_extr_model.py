import os, sys
import numpy as np
import json
import glob
import h5py
from numpyencoder import NumpyEncoder

from analyzer.model.utils.extracting import *
from analyzer.model.utils.helper import convert_dict_mtx

class FeatureExtractor():
    '''
    Using this model to build up your feature matrix that will be clustered.
    :param emvol & gtvol: (np.array) Both are the data volumes.
    :param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
                This is important as you might face memory problems loading the whole dataset into your RAM. Distinguish between two setups:
                - 'full': This enables reading the whole stack at once. Or at least the 'chunk_size' you set.
                - 'iter': This iterates over each slice/image and extracts information one by one.
                          This might help you to process the whole dataset without running into memory error.
    '''
    def __init__(self, cfg, emvol=None, gtvol=None):
        self.cfg = cfg
        self.emvol = emvol
        self.gtvol = gtvol
        self.empath = self.cfg.DATASET.EM_PATH
        self.gtpath = self.cfg.DATASET.LABEL_PATH
        self.dprc = self.cfg.MODE.DPRC
        self.ff = self.cfg.DATASET.FILE_FORMAT

        if self.dprc == 'iter':
            self.emfns = sorted(glob.glob(self.empath + '*.' + self.ff))
            self.gtfns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
        else:
            self.emfns = None
            self.gtfns = None

    def get_fns(self):
        '''Funtion returns the attribute fns of the feature extractor.'''
        return self.emfns, self.gtfns

    def compute_seg_size(self):
        '''Extract the size of each mitochondria segment.
        :returns result_dict: (dict) where the label is the key and the size of the segment is the corresponding value.
        '''
        return compute_region_size(self.gtvol, fns=self.gtfns, dprc=self.dprc)

    def compute_seg_slength(self):
        '''Extract the skeleton length of each mitochondria segment.
        :returns result_dict: (dict) where the label is the key and the size of the segment is the corresponding value.
        '''
        return compute_skeleton(fns=self.gtfns)

    def compute_seg_dist(self):
        '''Compute the distances of mitochondria to each other and extract it as a graph matrix.'''
        return compute_dist_graph(self.gtvol, fns=self.gtfns, dprc=self.dprc)

    def compute_vae_shape(self):
        '''
        Retrieves the shape feature generated by the variational autoencoder.
        :returns latent shape features for every object
        '''
        with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'shapef.h5'), 'r') as f:
            return f['shape']

    def compute_vae_texture(self):
        '''
        Retrieves the texture feature generated by the variational autoencoder.
        :returns latent texture features for every object
        '''
        with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'texturef.h5'), 'r') as f:
            return f['texture']

    def compute_vae_ptc_shape(self):
        '''
        Retrieves the shape feature generated by the variational autoencoder using pointclouds.
        :returns latent shape features for every object
        '''
        with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'ptc_shapef.h5'), 'w') as f:
            return f['ptc_shape']

    def compute_cl_shape(self):
        '''
        Retrieves the shape feature generated by the contrastive learning setup.
        :returns latent shape features for every object
        '''
        with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'clf.h5'), 'r') as f:
            return f['shape']

    def compute_seg_circ(self):
        '''Computes the circularity features from mitochondria volume.'''
        return compute_circularity(self.gtvol, fns=self.gtfns, dprc=self.dprc)

    def compute_seg_surface(self):
        '''Computes the surface to volume ratio features from mitochondria volume.'''
        return compute_surface(self.gtfns, self.cfg)

    def save_single_feat_h5(self, rsl_dict, filen='feature_vector'):
        '''
        Saving the resulting array dicts to h5 that contains the single features.
        :param rsl_dict: (array) of (dict)s that contains single features.
        :param filen: (string) filename. (without h5)
        '''
        labels, values = convert_dict_mtx(rsl_dict, filen[:-1])
        with h5py.File(self.cfg.DATASET.ROOTF + filen + '.h5', 'w') as h5f:
            h5f.create_dataset('id', data=labels)
            h5f.create_dataset(filen[:-1], data=values)

        print('saved features to {}.'.format(self.cfg.DATASET.ROOTF + filen + '.h5'))
        return labels, values

    def save_feats_h5(self, labels, feat_array, filen='features'):
        '''
        Saving arrays to h5 that contains the features.
        :param feat_array: that contains features.
        :param filen: (string) filename. (without h5)
        '''
        with h5py.File(self.cfg.DATASET.ROOTF + filen + '.h5', 'w') as h5f:
            h5f.create_dataset('id', data=labels)
            h5f.create_dataset(filen, data=feat_array)
        print('saved features to {}.'.format(self.cfg.DATASET.ROOTF + filen + '.h5'))

    def save_feat_dict(self, rsl_dict, filen='feature_vector.json'):
        '''
        Saving dict that contains the features to the designated folder.
        :param rsl_dict: (dict) that contains features.
        :param filen: (string) filename.
        '''
        with open(os.path.join(self.cfg.DATASET.ROOTF + filen), 'w') as f:
            json.dump(rsl_dict, f, cls=NumpyEncoder)
            f.close()
        print('stored infos in {}.'.format(os.path.join(self.cfg.DATASET.ROOTF + filen)))

    def compute_seg_spatial_density(self, n_block=10):
        return compute_spatial_density(fns=self.gtfns, n_block=n_block)
