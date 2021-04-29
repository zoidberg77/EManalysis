import os, sys
import numpy as np
import multiprocessing
import functools
import imageio
from scipy import signal
from skimage import measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import h5py

from analyzer.data.ptc_dataset import normalize_ptc

def point_cloud(fns, cfg, save=True):
	'''
	Calculating a point cloud representation for every segment in the Dataset.
	:param fns: (list) of images within the dataset.
	:param save: (Bool) Save it or not.
	:returns result_array: An (np.array) full of (dict)s that contain id and pts.
	'''
	print('Starting to compute the point representation of {} segments.'.format(len(fns)))
	result_dict = {}

	with multiprocessing.Pool(processes=cfg.SYSTEM.NUM_CPUS) as pool:
		tmp = pool.starmap(calc_point_repr, enumerate(fns))

	for dicts in tmp:
		for key, value in dicts.items():
			if key in result_dict:
				result_dict[key][0] = np.vstack((result_dict[key][0], value[0]))
			else:
				result_dict.setdefault(key, [])
				result_dict[key].append(value[0])

	if save:
		with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'w') as h5f:
			grp = h5f.create_group('ptcs')
			for result in result_dict.keys():
				std_rs = normalize_ptc(result_dict[result][0])
				grp.create_dataset(str(result), data=normalize_ptc(std_rs))
			h5f.close()
		print('saved point representations to {}.'.format(cfg.DATASET.ROOTD + 'vae/pts' + '.h5'))
	print('point cloud generation finished.')

def calc_point_repr(idx, fns):
	'''
	Helper for calculating the point representation.
	:param idx: This indicates the index of the image --> iterates therefor over the whole dataset.
	:param fns: This is a concrete filename.
	'''
	result = {}
	if os.path.exists(fns):
		tmp = imageio.imread(fns)
		regions = regionprops(tmp, cache=False)

		labels = list()
		cont_list = list()
		for props in regions:
			labels.append(props.label)
			seg = tmp.copy().astype(int)
			seg[tmp != props.label] = 0

			cs = measure.find_contours(seg, 0.8)
			cs3d = np.hstack((cs[0], np.full((cs[0].shape[0], 1), idx, dtype=cs[0].dtype)))
			cont_list.append(cs3d)

		for l in range(len(labels)):
			if labels[l] == 0:
				continue
			result.setdefault(labels[l], [])
			result[labels[l]].append(cont_list[l])

	return result


### Additional stuff here.
def get_surface_voxel(seg):
	assert seg.ndim == 3
	kernel = np.array([-1, 1])
	k_size = [1,1,1]
	seg = seg.copy().astype(int)
	surface = np.zeros_like(seg)
	indices, counts = np.unique(seg, return_counts = True)
	if indices[0] == 0:
		indices, counts = indices[1:], counts[1:]
	for i in range(3):
		temp_k_size = k_size.copy()
		temp_k_size[i] = 2
		temp_kernel = kernel.reshape(tuple(temp_k_size))
		temp = seg.copy()
		edge = convolve(temp, temp_kernel, mode='constant')
		surface += (edge!=0).astype(int)
	seg = seg[::-1, ::-1, ::-1]
	surface[seg == 0] = 0
	surface = (surface!=0).astype(int)
	return surface
