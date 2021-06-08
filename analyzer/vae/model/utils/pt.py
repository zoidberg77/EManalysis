import glob
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
from sklearn.preprocessing import normalize
from tqdm import tqdm

from analyzer.data.ptc_dataset import normalize_ptc

def point_cloud(cfg, dl, save=True):
	'''
	Calculating a point cloud representation for every segment in the Dataset.
	:param fns: (list) of images within the dataset.
	:param save: (Bool) Save it or not.
	:returns result_array: An (np.array) full of (dict)s that contain id and pts.
	'''
	_, fns, _ = dl.get_fns()
	print('Starting to compute the point representation of {} segments.'.format(len(fns)))
	result_dict = {}

	with multiprocessing.Pool(processes=cfg.SYSTEM.NUM_CPUS) as pool:
		tmp = list(tqdm(pool.imap(calc_point_repr, enumerate(fns)), total=len(fns)))

	for dicts in tmp:
		for key, value in dicts.items():
			if key in result_dict:
				result_dict[key][0] = np.vstack((result_dict[key][0], value[0]))
			else:
				result_dict.setdefault(key, [])
				result_dict[key].append(value[0])

	if save:
		with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'w') as h5f:
			h5f.create_dataset('labels', data=list(result_dict.keys()))
			grp = h5f.create_group('ptcs')
			for result in result_dict.keys():
				std_rs = normalize_ptc(result_dict[result][0])
				grp.create_dataset(str(result), data=std_rs)
			h5f.close()
		print('saved point representations to {}.'.format(cfg.DATASET.ROOTD + 'vae/pts' + '.h5'))
	print('point cloud generation finished.')

def calc_point_repr(fns):
	'''
	Helper for calculating the point representation.
	:param idx: This indicates the index of the image --> iterates therefor over the whole dataset.
	:param fns: This is a concrete filename.
	'''
	idx, fns = fns
	result = {}
	if idx >= 2:
		return result
	if os.path.exists(fns):
		tmp = imageio.imread(fns)
		regions = regionprops(tmp, cache=False)

		labels = list()
		cont_list = list()
		for props in regions:
			labels.append(props.label)
			#print('label of ptc: ', props.label)
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


# Additional stuff here.
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

'''
def generate_volume_ptc(cfg, dl):
	_, fns, _ = dl.get_fns()
	fns = fns[:50]
	ptcs = {}
	print("generating pointclouds for all {} slices".format(len(fns)))
	with multiprocessing.Pool(processes=cfg.SYSTEM.NUM_CPUS) as pool:
		ptcs_list = list(tqdm(pool.imap(get_coords_from_slice, enumerate(fns)), total=len(fns)))

	for ptc_item in ptcs_list:
		for key in ptc_item.keys():
			if key not in ptcs.keys():
				ptcs[key] = ptc_item[key]
			else:
				np.concatenate((ptcs[key], ptc_item[key]), axis=0)

	print("finished calculating point clouds")
	print("writing point clouds {} to h5".format(len(ptcs.keys())))
	with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'w') as h5f:
		h5f.create_dataset('labels', data=[key for key in ptcs.keys()])
		grp = h5f.create_group('ptcs')
		for key in ptcs.keys():
			coords = normalize_ptc(ptcs[key])
			grp.create_dataset(str(key), data=coords)
	print("finished writing point clouds to h5")
	return



def get_coords_from_slice(fn):
	z, fn = fn
	slice = imageio.imread(fn)
	ptcs = {}
	for id in np.unique(slice):
		if id == 0:
			continue
		coords = [[coord[0], coord[1], z] for coord in zip(*np.where(slice == id))]
		ptcs[str(id)] = coords
	return ptcs
'''
def generate_volume_ptc(cfg, dl):
	_, fns, _ = dl.get_fns()
	print("generating point clouds from {} slices".format(len(fns)))
	ptcs = {}
	for z, fn in tqdm(enumerate(fns), total=len(fns)):
		slice = imageio.imread(fn)
		objs = np.unique(slice)
		for obj in objs:
			if obj == 0:
				continue
			if str(obj) not in ptcs.keys():
				ptcs[str(obj)] = []
			ptcs[str(obj)] += [[coords[0], coords[1], z] for coords in zip(*np.where(slice == obj))]
	print("finished calculating point clouds")
	print("writing point clouds {} to h5".format(len(ptcs.keys())))
	with h5py.File(cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'w') as h5f:
		h5f.create_dataset('labels', data=[key for key in ptcs.keys()])
		grp = h5f.create_group('ptcs')
		for key in ptcs.keys():
			coords = ptcs[key]
			coords = normalize_ptc(coords)
			#x_min = np.min([c[0] for c in coords])
			#y_min = np.min([c[1] for c in coords])
			#z_min = np.min([c[2] for c in coords])
			#coords = [[c[0]-x_min, c[1]-y_min, c[2]-z_min] for c in coords]
			grp.create_dataset(str(key), data=coords)
	print("finished writing point clouds to h5")


'''
	
class PtcGenerator:
	def __init__(self, cfg, dl):
		self.cfg = cfg
		_, fns, _ = dl.get_fns()
		self.fns = fns
		self.dl = dl

	def generate_volume_ptc(self):
		print("loading object information")
		objs = self.dl.prep_data_info()
		print("generating {} point clouds".format(len(objs)))
		with h5py.File(self.cfg.DATASET.ROOTD + 'vae/pts' + '.h5', 'w') as h5f:
			h5f.create_dataset('labels', data=[obj['id'] for obj in objs])
			grp = h5f.create_group('ptcs')
			with multiprocessing.Pool(processes=self.cfg.SYSTEM.NUM_CPUS) as pool:
				results = list(tqdm(pool.imap(self.get_norm_coords, objs), total=len(objs)))
				print(len(results))

	def get_norm_coords(self, obj):
		id = obj['id']
		slices = obj['slices']
		volume = imageio.imread(self.fns[slices[0]])
		for slice in slices[1:]:
			volume = np.dstack((volume, imageio.imread(self.fns[slice])))
		coords = np.array([list(coord) for coord in zip(*np.where(volume == id))])
		norm_coords = normalize_ptc(coords)
		return [id, norm_coords]
'''