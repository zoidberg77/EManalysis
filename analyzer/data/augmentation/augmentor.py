import numpy as np
import torchvision.transforms as T
from analyzer.data.augmentation.composition import Compose
from analyzer.data.augmentation.rotation import Rotate
from analyzer.data.augmentation.color_jitter import ColorJitter
from analyzer.data.augmentation.flip import Flip
from analyzer.data.augmentation.cutblur import CutBlur
from analyzer.data import PtcDataset

class Augmentor():
    '''
        Augmentor object handles the augmentation of various input.
        :params cfg: (yacs.config.CfgNode): YACS configuration options.
    '''
    def __init__(self, volume_size, mean_std=0.5):
        self.volume_size = volume_size
        self.mean_std = mean_std
        self.aug_list = self.define_augmentation_op()
        self.transform = Compose(transforms=self.aug_list)

    def __call__(self, volume):
        x1 = np.expand_dims(self.transform(volume), axis=0).copy()
        x2 = np.expand_dims(self.transform(volume), axis=0).copy()
        return x1, x2

    def define_augmentation_op(self):
        aug_list = list()
        aug_list.append(Rotate(rot90=True, p=1.0))
        aug_list.append(ColorJitter((0, 0.4), (0, 0.4), (0, 0.4), (0, 0.1)))
        aug_list.append(Flip(do_ztrans=0))
        return aug_list

class PTCAugmentor():
    '''
        Augmentor object handles the augmentation of various input.
        Designed for point clouds.
        :params cfg: (yacs.config.CfgNode): YACS configuration options.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        ptcdl = PtcDataset(cfg)
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
        # visptc(rotated_data)
        # visptc(single_ptc.reshape((-1, 3)))
        # rotated_data = rotated_data[None, :, :]
        # print(rotated_data)
        # print(rotated_data.shape)

        return rotated_data
