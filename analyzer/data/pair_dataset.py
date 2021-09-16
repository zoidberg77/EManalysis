import numpy as np
import glob
import random
import torch
import torch.utils.data
from analyzer.data.utils.data_raw import *
from analyzer.data.utils.data_misc import *
from analyzer.data.augmentation import Augmentor

class PairDataset():
    '''
    This Dataloader will prepare sample that are pairs for feeding the contrastive
    learning algorithm.
    '''
    def __init__(self, cfg, iter_num: int = -1):
        self.cfg = cfg
        self.chunks_path = self.cfg.SSL.USE_PREP_DATASET
        self.sample_volume_size = (64, 64, 64)
        self.sample_stride = (1, 1, 1)
        self.cl_mode = self.cfg.MODE.PROCESS.replace('cl', '')
        self.augmentor = Augmentor(self.sample_volume_size)

        # Data information if you want to produce input on the fly.
        if not self.cfg.SSL.USE_PREP_DATASET:
            self.volume, self.label = self.get_input()
            self.volume_size = [np.array(self.volume.shape)]
            self.sample_volume_size = np.array(self.sample_volume_size).astype(int)
            self.sample_stride = np.array(self.sample_stride).astype(int)
            self.sample_size = [count_volume(self.volume_size[x], self.sample_volume_size, self.sample_stride)
                                for x in range(len(self.volume_size))]
            self.sample_num = np.array([np.prod(x) for x in self.sample_size])
            self.sample_num_a = np.sum(self.sample_num)
            self.sample_num_c = np.cumsum([0] + list(self.sample_num))

            self.iter_num = max(iter_num, self.sample_num_a)
            print('Dataset chunks that will be iterated over: {}'.format(self.iter_num))

    def __len__(self):
        if not self.cfg.SSL.USE_PREP_DATASET:
            return self.iter_num
        else:
            with h5py.File(self.chunks_path, 'r') as f:
                return len(f['id'])

    def __getitem__(self, idx):
        return self.create_sample_pair(idx)

    def create_sample_pair(self, idx):
        '''Create a sample pair that will be used for contrastive learning.
        '''
        if not self.cfg.SSL.USE_PREP_DATASET:
            sample = self.reject_sample()
        else:
            with h5py.File(self.chunks_path, 'r') as f:
                sample = f['chunk'][idx]
                unique_label = int(f['id'][idx])
                if 'gt' in list(f.keys()):
                    gt_label = int(f['gt'][idx])
                else:
                    gt_label = None
                if sample.ndim > 3:
                    sample = np.squeeze(sample)
        if self.cl_mode == 'train':
            sample_pair = self.augmentor(sample)
            return (sample_pair, unique_label, gt_label)
        else:
            return (np.expand_dims(sample, axis=0).copy(), unique_label, gt_label)

    def create_chunk_volume(self):
        '''
        Function creates small chunk from input volume that is processed
        into the training model.
        '''
        pos = self.get_pos(self.sample_volume_size)
        pos, out_vol, out_label = self.crop_with_pos(pos, self.sample_volume_size)
        return pos, self.create_masked_input(out_vol, out_label)

    def create_masked_input(self, vol: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Create masked input volume, that is pure EM where the mask is not 0. Otherwise all
        values set to 0. Returns the prepared mask.
        :params vol (numpy.ndarray): volume that is EM input.
        :params label (numpy.ndarray): associated label volume.
        '''
        vol[np.where(label == 0)] = 0
        return np.array(vol)

    def get_input(self):
        '''Get input volume and labels.'''
        emfns = sorted(glob.glob(self.cfg.DATASET.EM_PATH + '*.' + self.cfg.DATASET.FILE_FORMAT))
        labelfns = sorted(glob.glob(self.cfg.DATASET.LABEL_PATH + '*.' + self.cfg.DATASET.FILE_FORMAT))
        if len(emfns) == 1:
            vol = readvol(emfns[0])
            label = readvol(labelfns[0])
        else:
            vol = folder2Vol(chunk_size=self.cfg.DATASET.CHUNK_SIZE, fns=emfns, file_format=self.cfg.DATASET.FILE_FORMAT)
            label = folder2Vol(chunk_size=self.cfg.DATASET.CHUNK_SIZE, fns=labelfns, file_format=self.cfg.DATASET.FILE_FORMAT)

        return vol, label

    def crop_with_pos(self, pos, vol_size):
        out_volume = (crop_volume(
            self.volume, vol_size, pos[1:])/255.0).astype(np.float32)
        out_label = crop_volume(
            self.label, vol_size, pos[1:])
        return pos, out_volume, out_label

    def get_pos(self, vol_size):
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self.index_to_dataset(random.randint(0, self.sample_num - 1))
        pos[0] = did
        # pick a position
        tmp_size = count_volume(
            self.volume_size[did], vol_size, self.sample_stride)
        tmp_pos = [random.randint(0, tmp_size[x]-1) * self.sample_stride[x]
                   for x in range(len(tmp_size))]

        pos[1:] = tmp_pos
        return pos

    def index_to_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1

    def reject_sample(self):
        '''function makes sure that sample contains actual objects that are
        sufficiently large enough.'''
        while True:
            _, sample = self.create_chunk_volume()
            if np.count_nonzero(sample) > 0:
                return sample
