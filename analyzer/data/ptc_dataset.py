import os, sys
import numpy as np
import h5py
from sklearn.preprocessing import normalize

def normalize_ptc(ptc):
    '''
    Function normalizes the ptc (Nxd) by min-max-scaling.
    :param ptc: (np.ndarray) size: Nxd
    '''
    return normalize(ptc, axis=0, norm='max')

def rotate_point_cloud(batch_data):
    '''
    Randomly rotate the point clouds to augument the dataset rotation is per shape based along up direction
    '''
    #TODO
    '''
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data
    '''
    pass

class PtcDataloader():
    '''
    This is the Data module for the pointcloud autoencoder.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.ptfn = cfg.DATASET.ROOTD + 'vae/pts' + '.h5'

    def __len__(self):
        '''
        Required by torch to return the length of the dataset.
        :returns: integer
        '''
        with h5py.File(self.ptfn, 'r') as h5f:
            return len(list(h5f.get('ptcs').keys()))

    def __getitem__(self, idx):
        '''
        Required by torch to return one item of the dataset.
        :param idx: (int) index of the object. Please note that this is the actual label e.g. 1325 not a pure index like 0,1,2, ... ,n
        :returns: object from the volume. (np.array)
        '''
        with h5py.File(self.ptfn, 'r') as h5f:
            group = h5f.get('ptcs')
            tmp = np.array(group[str(idx)])
            return tmp[None,:,:]

    @property
    def keys(self):
        '''property that gives to a list of keys (ints) that are in the dataset.
        '''
        with h5py.File(self.ptfn, 'r') as h5f:
            return list(map(int, list(h5f.get('ptcs').keys())))
