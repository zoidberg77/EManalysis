import os, sys
import numpy as np
import multiprocessing
import functools
import imageio
from scipy import signal
from skimage import measure
from skimage.measure import label, regionprops

def point_cloud(fns):
	'''
	Calculating a point cloud representation for every segment in the Dataset.
	:param fns: (list) of images within the dataset.
	'''
	print('Starting to compute the point representation of every segments.')
	result_dict = {}

	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		tmp = pool.starmap(calc_point_repr, enumerate(fns))

	for dicts in tmp:
		for key, value in dicts.items():
			if key in result_dict:
				result_dict[key][0] += value[0]
			else:
				result_dict.setdefault(key, [])
				result_dict[key].append(value[0])

	result_array = []
	for result in result_dict.keys():
		result_array.append({
			'id': result,
			'coords': result_dict[result],
		})
	print('Size point cloud generation finished. {} features extracted.'.format(len(result_array)))
	return (result_array)

def calc_point_repr(idx, fns):
	'''
	Helper for calculating the point representation.
	'''
	result = {}
	if os.path.exists(fns):
		tmp = imageio.imread(fns)
		regions = regionprops(tmp, cache=False)
		#cs = measure.find_contours(tmp, 0.5)

		labels = []
		coord_list = []
		for props in regions:
			labels.append(props.label)
			coord_list.append(props.coords)

		for l in range(len(labels)):
			if labels[l] == 0:
				continue
			result.setdefault(labels[l], [])
			result[labels[l]].append(coord_list[l])

	return result

def get_surface_voxel(seg):
	'''
	Convert a voxel representation to a surface which consists of single points.
	:param seg:
	'''
	if seg.ndim == 3:
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
			edge = signal.convolve(temp, temp_kernel, mode='same')
			surface += (edge!=0).astype(int)
		seg = seg[::-1, ::-1, ::-1]
		surface[seg == 0] = 0
		surface = (surface!=0).astype(int)
	else:
		kernel = np.array([-1, 1])
		k_size = [1,1]
		seg = seg.copy().astype(int)
		surface = np.zeros_like(seg)
		indices, counts = np.unique(seg, return_counts = True)
		if indices[0] == 0:
			indices, counts = indices[1:], counts[1:]
		for i in range(2):
			temp_k_size = k_size.copy()
			temp_k_size[i] = 2
			temp_kernel = kernel.reshape(tuple(temp_k_size))
			temp = seg.copy()
			edge = signal.convolve(temp, temp_kernel, mode='same')
			surface += (edge!=0).astype(int)
		seg = seg[::-1, ::-1]
		surface[seg == 0] = 0
		surface = (surface!=0).astype(int)
	return surface
