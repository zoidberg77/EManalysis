import os, sys
import random

import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import multiprocessing as mp

def normalize_ptc(ptc):
    '''
    Function normalizes the ptc (Nxd) by min-max-scaling.
    :param ptc: (np.ndarray) size: Nxd
    '''
    return normalize(ptc, axis=0, norm='max')

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
                    # clouds = [(str(key), np.array(cloud)) for key, cloud in h5f['ptcs'].items()]
                    # #clouds = clouds[:10]
                    # pool = mp.Pool(processes=cfg.SYSTEM.NUM_CPUS)
                    # results = [pool.apply(self.calculate_blue_noise_samples, args=(key, cloud,)) for key, cloud in
                    #            tqdm(clouds, total=len(clouds))]
                    # with h5py.File(self.rptcfn, 'w') as random_points_file:
                    #     for result in results:
                    #         random_points_file[result[0]] = result[1]
                    with h5py.File(self.sampled_ptfn, 'w') as random_points_file:
                        for k, c in tqdm(list(h5f['ptcs'].items()), total=len(h5f['ptcs'].items())):
                            key = str(k)
                            cloud = np.array(c)
                            key, random_points = self.calculate_blue_noise_samples(key, cloud)
                            random_points_file[key] = random_points

    def calculate_blue_noise_samples(self, key, cloud):
        '''helper for calculating blue noise.'''
        idxs = []
        dists = pairwise_distances(cloud)
        possible_idx = list(np.arange(0, len(cloud)))
        start = random.sample(possible_idx, 1)[0]
        idxs.append(start)
        for i in range(1, self.sample_size):
            bnsp = self.blue_noise_sample_points
            if bnsp > len(possible_idx):
                bnsp = len(possible_idx)
            candidates = np.random.randint(0, len(cloud), bnsp)
            reduced_dists = dists[candidates, :][:, idxs]
            sums = np.sum(reduced_dists, axis=1)
            best_candidate = candidates[np.argmax(sums)]
            idxs.append(best_candidate)
        random_points = cloud[idxs, :]
        return key, random_points

    def __len__(self):
        '''Required by torch to return the length of the dataset. Returns: (int).'''
        if self.sample_mode == 'partial':
            with h5py.File(self.ptfn, 'r') as h5f:
                return len(list(h5f.get('ptcs').keys()))
        elif self.sample_mode is not None:
            #with h5py.File(self.rptcfn, 'r') as random_points_file:
            with h5py.File(self.sampled_ptfn, 'r') as random_points_file:
                return len(random_points_file.keys())
        else:
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
            idx = sorted(list(group.keys()))[idx]
            ptc = np.array(group[idx])
        if self.sample_mode == 'partial':
                if ptc.shape[0] > self.sample_size:
                    randome_indices = np.random.random_integers(ptc.shape[0] - 1, size=(self.sample_size))
                    return np.expand_dims(ptc[randome_indices, :], axis=0), idx
                return np.expand_dims(ptc, axis=0), idx
        elif self.sample_mode is not None:
            #with h5py.File(self.rptcfn, 'r') as random_points_file:
            with h5py.File(self.sampled_ptfn, 'r') as random_points_file:
                return np.expand_dims(random_points_file[str(idx)], axis=0), idx
        else:
            return np.expand_dims(ptc, axis=0), idx

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
