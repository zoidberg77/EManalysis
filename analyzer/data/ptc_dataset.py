import os, sys
import random

import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
from sklearn.metrics import euclidean_distances
import multiprocessing as mp
from torchvision import transforms
import torch


def normalize_ptc(ptc):
    '''
    Function normalizes the ptc (Nxd) by min-max-scaling.
    :param ptc: (np.ndarray) size: Nxd
    '''
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(ptc)

class PtcDataset():
    '''
    This is the Data module for the pointcloud autoencoder.
    '''
    def __init__(self, cfg, sample_size=2000, sample_mode=None):
        self.cfg = cfg
        self.sample_size = self.cfg.PTC.SAMPLE_SIZE
        self.ptfn = self.cfg.PTC.INPUT_DATA
        self.sample_mode = self.cfg.PTC.SAMPLE_MODE
        self.dists = {}
        self.blue_noise_sample_points = cfg.PTC.BLUE_NOISE_SAMPLE_POINTS
        self.sampled_ptfn = self.cfg.PTC.INPUT_DATA_SAMPLED
        self.rptcfn = cfg.DATASET.ROOTD + 'vae/random_ptc' + '.h5'

        if self.sample_mode == 'whitenoise':
            if os.path.exists(self.rptcfn) or os.path.exists(self.sampled_ptfn):
                print('{} exists and will be used.'.format(self.sampled_ptfn))
            else:
                print("Calculating random points via white noise sampling.")
                with h5py.File(self.ptfn, 'r') as h5f:
                    #with h5py.File(self.rptcfn, 'w') as random_points_file:
                    with h5py.File(self.sampled_ptfn, 'w') as random_points_file:
                        for key, cloud in tqdm(h5f['ptcs'].items(), total=len(h5f['ptcs'].keys())):
                            idxs = np.random.randint(0, len(cloud), self.sample_size)
                            random_points_file[key] = np.array(cloud)[idxs, :].astype(np.double)

        if self.sample_mode == 'montecarlo':
            if os.path.exists(self.rptcfn) or os.path.exists(self.sampled_ptfn):
                print('{} exists and will be used.'.format(self.sampled_ptfn))
            else:
                print("Calculating random points via monte carlo (MC) sampling")
                with h5py.File(self.ptfn, 'r') as h5f:
                    #with h5py.File(self.rptcfn, 'w') as random_points_file:
                    with h5py.File(self.sampled_ptfn, 'w') as random_points_file:
                        group = h5f.get('ptcs')
                        for idx in tqdm(group.keys(), total=len(group.keys())):
                            cloud = np.array(group[idx])
                            centroid = np.mean(cloud, axis=0)
                            dists = []
                            for point in cloud:
                                dists.append(np.linalg.norm(point - centroid))
                            dists /= sum(dists)
                            xk = np.arange(len(dists))
                            custm = stats.rv_discrete(name='custm', values=(xk, dists))
                            random_points = cloud[custm.rvs(size=self.sample_size), :]
                            random_points_file[idx] = random_points

        if self.sample_mode == "bluenoise":
            if os.path.exists(self.rptcfn) or os.path.exists(self.sampled_ptfn):
                print('{} exists and will be used.'.format(self.sampled_ptfn))
            else:
                print("calculating random points via bluenoise sampling")
                with h5py.File(self.ptfn, 'r') as h5f:
                    with h5py.File(self.sampled_ptfn, 'w') as random_points_file:
                        random_points_file.create_dataset("labels", shape=(len(list(h5f['ptcs'].keys())), ))
                        random_points_file.create_dataset("ptcs", shape=(len(list(h5f['ptcs'].keys())), cfg.PTC.SAMPLE_SIZE, 3))
                        i = 0
                        for key in tqdm(list(h5f['ptcs'].keys())):
                            cloud = np.array(h5f['ptcs'][key])
                            key, random_points = self.calculate_blue_noise_samples(key, cloud)
                            random_points_file['labels'][i] = key
                            random_points_file['ptcs'][i] = random_points
                            i += 1

    def calculate_blue_noise_samples(self, key, cloud):
        '''helper for calculating blue noise.'''
        idxs = []
        possible_idx = list(np.arange(0, len(cloud)))
        start = random.sample(possible_idx, 1)[0]
        idxs.append(start)
        for i in range(1, self.sample_size):
            bnsp = self.blue_noise_sample_points
            if bnsp > len(possible_idx):
                bnsp = len(possible_idx)
            candidates = np.random.randint(0, len(cloud), bnsp)
            reduced_dists = euclidean_distances(cloud[candidates, :], cloud[idxs, :])
            sums = np.sum(reduced_dists, axis=1)
            best_candidate = candidates[np.argmax(sums)]
            idxs.append(best_candidate)
        random_points = cloud[idxs, :]
        return key, random_points

    def __len__(self):
        '''Required by torch to return the length of the dataset. Returns: (int).'''
        with h5py.File(self.sampled_ptfn, 'r') as random_points_file:
            return len(random_points_file["ptcs"])



    def __getitem__(self, idx):
        '''
        Required by torch to return one item of the dataset.
        :param idx: (int) index of the object. Please note that this is NOT the actual label.
        :returns: object from the volume. (np.array)
        '''

        with h5py.File(self.sampled_ptfn, 'r') as random_points_file:
            points = np.array(random_points_file["ptcs"][idx], dtype=np.float32)
            return torch.from_numpy(points), random_points_file["labels"][idx]


    @property
    def keys(self):
        '''property that gives to a list of keys (ints) that are in the dataset.
        '''
        with h5py.File(self.ptfn, 'r') as h5f:
            return list(map(int, list(h5f.get('ptcs').keys())))

    @property
    def dimlist(self):
        '''returns list of number of points that every point cloud contains.'''
        dim_list = list()
        with h5py.File(self.ptfn, 'r') as h5f:
            group = h5f.get('ptcs')
            for _, idx in enumerate(list(h5f.get('ptcs').keys())):
                dim_list.append(np.array(group[str(idx)]).shape[0])
        return dim_list

    def split_dataset(self):
        '''split dataset and keep order (avoid loss of label information).'''
        pass

    def recur(self, group, idx):
        '''helper to overcome a missed label'''
        if str(idx) in list(group.keys()):
            tmp = np.array(group[str(idx)])
            return tmp, idx
        else:
            idx = idx + 1
            tmp, idx = self.recur(group, idx)
            return tmp, idx

    def save_cloud_vis(self, cloud, random_cloud):
        pass
