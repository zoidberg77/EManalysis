import numpy as np
from analyzer.data import PtcDataset

class Augmentor():
    '''
        Augmentor object handles the augmentation of various input.
        :params cfg: (yacs.config.CfgNode): YACS configuration options.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        ptcdl = PtcDataset(cfg)
        print(ptcdl[0][0].shape)
        self.rotate_point_cloud(ptcdl[0][0])

    def rotate_point_cloud(self, single_ptc):
        '''Randomly rotate the point clouds to augument the dataset.
        Rotation is per shape based along up direction.
        '''
        rotated_data = np.zeros(single_ptc.shape, dtype=np.float32)

        rotation_angle = np.random.uniform() * 2 * np.pi

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])

        rotated_data = np.dot(single_ptc.reshape((-1, 3)), rotation_matrix)
        rotated_data = rotated_data[None, :, :]
        print(rotated_data)
        print(rotated_data.shape)

        # for k in xrange(batch_data.shape[0]):
        #     rotation_angle = np.random.uniform() * 2 * np.pi
        #     cosval = np.cos(rotation_angle)
        #     sinval = np.sin(rotation_angle)
        #     rotation_matrix = np.array([[cosval, 0, sinval],
        #                                 [0, 1, 0],
        #                                 [-sinval, 0, cosval]])
        #     shape_pc = batch_data[k, ...]
        #     rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        # return rotated_data
