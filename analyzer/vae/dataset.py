import glob
import multiprocessing
import os

import h5py
import imageio
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.transform import resize
from tqdm import tqdm

import analyzer.data


class MitoDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.region_limit = cfg.AUTOENCODER.REGION_LIMIT
        self.chunks_per_cpu = cfg.AUTOENCODER.CHUNKS_CPU
        self.upper_limit = cfg.AUTOENCODER.UPPER_BOUND
        self.lower_limit = cfg.AUTOENCODER.LOWER_BOUND
        self.mito_volume_file_name = cfg.AUTOENCODER.OUPUT_FILE_VOLUMES
        self.mito_volume_dataset_name = cfg.AUTOENCODER.DATASET_NAME
        self.gt_path = cfg.DATASET.LABEL_PATH
        self.em_path = cfg.DATASET.EM_PATH
        self.target_size = cfg.AUTOENCODER.TARGET
        self.ff = cfg.DATASET.FILE_FORMAT
        self.cpus = cfg.SYSTEM.NUM_CPUS

    def __len__(self):
        '''
        Required by torch to return the length of the dataset.
        :returns: integer
        '''
        with h5py.File(self.mito_volume_file_name, 'r') as f:
            return f[self.mito_volume_dataset_name].shape[0]

    def __getitem__(self, idx):
        '''
        Required by torch to return one item of the dataset.
        :param idx: index of the object
        :returns: object from the volume
        '''
        with h5py.File(self.mito_volume_file_name, 'r') as f:
            return f[self.mito_volume_dataset_name][idx]

    def extract_scale_mitos(self):
        dl = analyzer.data.Dataloader(self.cfg)
        regions = dl.prep_data_info()
        print("{} objects found in the ground truth".format(len(regions)))
        regions = pd.DataFrame(regions)
        regions = regions[(self.upper_limit > regions['size']) & (self.lower_limit < regions['size'])].values.tolist()
        filtered_length = len(regions)
        print("{} within limits {} and {}".format(filtered_length, self.lower_limit, self.upper_limit))
        if self.region_limit is not None:
            regions = regions[:self.region_limit]
            print("{} will be extracted due to set region_limit".format(self.region_limit))
        mode = 'w'
        start = 0
        '''
        if os.path.exists(self.mito_volume_file_name):
            mode = 'a'
        '''

        dset = None
        with h5py.File(self.mito_volume_file_name, mode) as f:
            if mode == 'w':
                dset = f.create_dataset(self.mito_volume_dataset_name, (
                    len(regions), 1, self.target_size[0], self.target_size[1], self.target_size[2]),
                                        maxshape=(None, 1, self.target_size[0], self.target_size[1],
                                                  self.target_size[2]))
            else:
                dset = f[self.mito_volume_dataset_name]
                for i, mito in enumerate(dset):
                    if np.max(mito) == 0:
                        start = i
                        print('found file with {} volumes in it'.format(i))
                        break

            with multiprocessing.Pool(processes=self.cpus) as pool:
                for i in tqdm(range(start, len(regions), int(self.cpus * self.chunks_per_cpu))):
                    results = pool.map(self.get_mito_volume, regions[i:i + int(self.cpus * self.chunks_per_cpu)])
                    for j, result in enumerate(results):
                        dset[i + j] = result

    def get_mito_volume(self, region):
        '''
        Preprocessing function to extract and scale the mitochondria as volume
        :param region: (dict) one region object provided by Dataloader.prep_data_info
        :returns result: (numpy.array) a numpy array with the target dimensions and the mitochondria in it
        '''
        all_fn = sorted(glob.glob(self.gt_path + '*.' + self.ff))
        fns = [all_fn[id] for id in region[2]]
        first_image_slice = imageio.imread(fns[0])
        mask = np.zeros(shape=first_image_slice.shape, dtype=np.uint16)
        mask[first_image_slice == region[0]] = 1
        volume = mask

        for fn in fns[1:]:
            image_slice = imageio.imread(fn)
            mask = np.zeros(shape=image_slice.shape, dtype=np.uint16)
            mask[image_slice == region[0]] = 1
            volume = np.dstack((volume, mask))
        volume = np.moveaxis(volume, -1, 0)

        mito_regions = regionprops(volume, cache=False)
        if len(mito_regions) != 1:
            print("something went wrong during volume building. region count: {}".format(len(mito_regions)))

        mito_region = mito_regions[0]

        mito_volume = volume[mito_region.bbox[0]:mito_region.bbox[3] + 1,
                      mito_region.bbox[1]:mito_region.bbox[4] + 1,
                      mito_region.bbox[2]:mito_region.bbox[5] + 1].astype(np.float32)

        scaled_mito = resize(mito_volume, self.target_size)
        scaled_mito = scaled_mito/scaled_mito.max()
        scaled_mito = np.expand_dims(scaled_mito, 0)

        return scaled_mito
