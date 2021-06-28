import numpy as np
import torchvision.transforms as T
from analyzer.data import PtcDataset
from analyzer.data.data_vis import visptc

class Augmentor():
    '''
        Augmentor object handles the augmentation of various input.
        :params cfg: (yacs.config.CfgNode): YACS configuration options.
    '''
    def __init__(self, volume_size, mean_std=0.5):
        self.volume_size = volume_size
        self.mean_std = mean_std
        self.transform = T.Compose([
            T.RandomResizedCrop(self.volume_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            #T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, volume):
        x1 = self.transform(volume)
        x2 = self.transform(volume)
        return x1, x2

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
