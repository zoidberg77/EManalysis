import glob
import multiprocessing
import os

import h5py
import imageio
import numpy as np
import torch
import torchio as tio
from skimage.measure import regionprops
from tqdm import tqdm

import analyzer.data


class MitoDataset:
    def __init__(self, em_path, gt_path, mito_volume_file_name="features/mito.h5",
                 mito_volume_dataset_name="mito_volumes",
                 target_size=(1, 64, 64, 64), lower_limit=100, upper_limit=100000, chunks_per_cpu=4, ff="png",
                 region_limit=None):
        self.region_limit = region_limit
        self.chunks_per_cpu = chunks_per_cpu
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.mito_volume_file_name = mito_volume_file_name
        self.mito_volume_dataset_name = mito_volume_dataset_name
        self.gt_path = gt_path
        self.em_path = em_path
        self.target_size = target_size
        self.ff = ff

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

    def get_mito_volume(self, region):
        '''
        Preprocessing function to extract and scale the mitochondria as volume
        :param region: (dict) one region object provided by Dataloader.prep_data_info
        :returns result: (numpy.array) a numpy array with the target dimensions and the mitochondria in it
        '''
        all_fn = sorted(glob.glob(self.gt_path + '*.' + self.ff))
        target = tio.ScalarImage(tensor=torch.rand(self.target_size))
        fns = [all_fn[id] for id in region['slices']]
        first_image_slice = imageio.imread(fns[0])
        mask = np.zeros(shape=first_image_slice.shape, dtype=np.uint16)
        mask[first_image_slice == region['id']] = 1
        volume = mask

        for fn in fns[1:]:
            image_slice = imageio.imread(fn)
            mask = np.zeros(shape=image_slice.shape, dtype=np.uint16)
            mask[image_slice == region['id']] = 1
            volume = np.dstack((volume, mask))
        volume = np.moveaxis(volume, -1, 0)

        mito_regions = regionprops(volume, cache=False)
        if len(mito_regions) != 1:
            print("something went wrong during volume building. region count: {}".format(len(mito_regions)))

        mito_region = mito_regions[0]
        mito_volume = volume[mito_region.bbox[0]:mito_region.bbox[3] + 1,
                      mito_region.bbox[1]:mito_region.bbox[4] + 1,
                      mito_region.bbox[2]:mito_region.bbox[5] + 1].astype(np.float32)
        mito_volume = np.expand_dims(mito_volume, 0)
        if self.lower_limit > mito_volume.sum() > self.upper_limit:
            return None

        transform = tio.Resample(target=target, image_interpolation='nearest')
        transformed_mito = transform(mito_volume)

        return transformed_mito

    def extract_scale_mitos(self,
                            cpus=multiprocessing.cpu_count()):
        dl = analyzer.data.Dataloader(gtpath=self.gt_path, volpath=self.em_path)
        regions = dl.prep_data_info()
        if self.region_limit is not None:
            regions = regions[:self.region_limit]
        print("{} mitochondira found in the ground truth".format(len(regions)))
        mode = 'w'
        start = 0
        if os.path.exists(self.mito_volume_file_name):
            mode = 'a'

        dset = None
        with h5py.File(self.mito_volume_file_name, mode) as f:
            if mode == 'w':
                dset = f.create_dataset(self.mito_volume_dataset_name, (
                    len(regions), self.target_size[0], self.target_size[1], self.target_size[2], self.target_size[3]),
                                        maxshape=(None, self.target_size[0], self.target_size[1], self.target_size[2],
                                                  self.target_size[3]))
            else:
                dset = f[self.mito_volume_dataset_name]
                for i, mito in enumerate(dset):
                    if np.max(mito) == 0:
                        start = i
                        print('found file with {} volumes in it'.format(i))
                        break

            with multiprocessing.Pool(processes=cpus) as pool:
                mito_counter = 0
                for i in tqdm(range(start, len(regions), cpus * self.chunks_per_cpu)):
                    results = pool.map(self.get_mito_volume, regions[i:i + cpus * self.chunks_per_cpu])
                    for j, result in enumerate(results):
                        if result is not None:
                            dset[i + j] = result
                            mito_counter += 1
                        print()
