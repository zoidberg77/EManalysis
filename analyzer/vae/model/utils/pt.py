import glob
import os, sys
from turtle import shape
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
from functools import partial

from analyzer.data.ptc_dataset import normalize_ptc

def point_cloud(cfg, dl, save=True):
	'''
	Calculating a point cloud representation for every segment in the Dataset.
	:param fns: (list) of images within the dataset.
	:param save: (Bool) Save it or not.
	:returns result_array: An (np.array) full of (dict)s that contain id and pts.
	'''
	_, fns, _ = dl.get_fns()
	print('Starting to compute the point representation extracted from {} images.'.format(len(fns)))
	results = []
	objs = dl.prep_data_info(save=True)
	get_coords = partial(get_coords_from_slices, fns=fns)
	with multiprocessing.Pool(cfg.SYSTEM.NUM_CPUS) as p:
		results = p.map(get_coords, objs)
	print("finished pointcloud calculation")
	with h5py.File(cfg.PTC.INPUT_DATA, 'w') as h5f:
		grp = h5f.create_group('ptcs')
		for k, result in tqdm(results):
			std_rs = normalize_ptc(result)
			grp.create_dataset(str(k), data=std_rs)

	print('saved point representations to {}.'.format(cfg.PTC.INPUT_DATA))
	print('point cloud generation finished.')

def get_coords_from_slices(region, fns):
	all_files = [fns[id] for id in region["slices"]]
	volume = []
	for img_file in all_files:
		slice = imageio.imread(img_file)
		slice[slice != region["id"]] = 0
		volume.append(slice)
	volume = np.array(volume)
	coords = np.transpose(np.nonzero(volume))
	return region["id"], coords

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
	with h5py.File(cfg.PTC.INPUT_DATA, 'w') as h5f:
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
