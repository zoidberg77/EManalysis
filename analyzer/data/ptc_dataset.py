import os, sys
import numpy as np
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
    def __init__(self):
        pass
