import os, sys
import json
import math
import numpy as np
import multiprocessing
import functools
import imageio
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial import distance
from analyzer.data.data_vis import visvol

def compute_region_size(vol, dprc='full', fns=None, mode='3d'):
	'''
	Compute the region properties of the groundtruth labels.

	:param vol: volume (np.array) that contains the labels. (2d || 3d)
	:param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
	:param fns: (list) list of filenames that should be used for iterating over.
	:param mode: (string)

	:returns result_array: (np.array) which contains (dicts) where the label is the key and the size of the segment is the corresponding value.
	'''
	print('Starting to extract size features.')
	result_dict = {}

	if dprc == 'full':
		labels = np.zeros(shape=vol.shape, dtype=np.uint16)
		label_cnt = 0
		if mode == '2d':
			### NOTE: For now this is depracted and will disappear in the future (probably).
			### Use 3d instead!
			for idx in range(vol.shape[0]):
				image = vol[idx, :, :]
				label2d, num_label = label(image, return_num=True)
				regions = regionprops(label2d, cache=False)

				for props in regions:
					area_values.append(props.area)

				tmp = np.zeros(shape=image.shape, dtype=np.uint16)
				if idx == 0:
					labels[idx, :, :] = label2d
				else:
					tmp = label2d
					tmp[tmp != 0] += label_cnt
					labels[idx, :, :] = tmp

				label_cnt += num_label

		if mode == '3d':
			if vol.ndim <= 2:
				raise ValueError('Volume is lacking on dimensionality(at least 3d): {}'.format(vol.shape))

			regions = regionprops(vol, cache=False)
			for props in regions:
				try_label = props.label
				area = props.area
				result_dict[try_label] = area

			result_array = []
			for result in result_dict.keys():
				result_array.append({
					'id': result,
					'size': result_dict[result],
				})

	elif dprc == 'iter':
		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			tmp = pool.starmap(functools.partial(calc_props, prop_list=['size']), enumerate(fns))

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
				'size': result_dict[result],
			})
	else:
		raise ValueError('No proper dprc found. Choose \'full\' or \'iter\'.')
	print('Size feature extraction finished. {} features extracted.'.format(len(result_array)))
	return (result_array)

def compute_dist_graph(vol, dprc='full', fns=None):
	'''
	This function computes a graph matrix that represents the distances from each segment to all others.
	:param vol: volume (np.array) that contains the groundtruth mask (= labels). (2d || 3d)
	:param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
	:returns: (np.array) (N x N) matrix gives you the feature vector--> N: number of segments
	'''
	print('Starting to compute distances between mitochondria.')
	result_dict = {}
	if dprc == 'full':
		if vol.ndim <= 2:
			raise ValueError('Volume is lacking on dimensionality(at least 3d): {}'.format(vol.shape))

		regions = regionprops(vol, cache=False)
		labels = []
		centerpts = []
		for props in regions:
			labels.append(props.label)
			centerpts.append(props.centroid)

		centerpts = np.array(centerpts, dtype=np.int16)
		dist_m = distance.cdist(centerpts, centerpts, 'euclidean')

		for idx in range(len(labels)):
			result_dict[labels[idx]] = dist_m[idx]

		result_array = []
		for result in result_dict.keys():
			result_array.append({
				'id': result,
				'dist': result_dict[result],
			})

	elif dprc == 'iter':
		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			tmp = pool.starmap(functools.partial(calc_props, prop_list=['size', 'slices', 'centroid']), enumerate(fns))

		for dicts in tmp:
			for key, value in dicts.items():
				if key in result_dict:
					result_dict[key][0].append(value[0])
					result_dict[key][1].append(value[1])
					result_dict[key][2].append(value[2])
				else:
					result_dict.setdefault(key, [])
					result_dict[key].append([value[0]])
					result_dict[key].append([value[1]])
					result_dict[key].append([value[2]])

		labels = list(result_dict.keys())
		centerpts = []
		for key, value in result_dict.items():
			pt = list(map(int, [sum(x) / len(x) for x in zip(*value[2])]))
			tmp_z = 0.0
			for i in range(len(value[0])):
				tmp_z += (value[0][i] / sum(value[0])) * value[1][i]
			z = int(tmp_z / len(value[1]))
			pt.append(z)
			centerpts.append(pt)

		centerpts = np.array(centerpts, dtype=np.int16)
		dist_m = distance.cdist(centerpts, centerpts, 'euclidean')

		for idx in range(len(labels)):
			result_dict[labels[idx]].append([dist_m[idx]])

		result_array = []
		for result in result_dict.keys():
			result_array.append({
				'id': result,
				'dist': result_dict[result][3],
			})
	else:
		raise ValueError('No proper dprc found. Choose \'full\' or \'iter\'.')
	print('Distance feature extraction finished. {} x {} features extracted.'.format(len(result_array), len(result_array)))
	return (result_array)

def compute_circularity(vol, dprc='full', fns=None):
	'''
	This function aims to calculate the circularity of an object.
	'''
	print('Starting to compute a circularity estimation of mitochondria.')
	result_dict = {}
	if dprc == 'full':
		fns = fns[:vol.shape[0]]

	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		tmp = pool.starmap(functools.partial(calc_props, prop_list=['slices', 'circ']), enumerate(fns))

	for dicts in tmp:
		for key, value in dicts.items():
			if key in result_dict:
				result_dict[key][0].append(value[0])
				result_dict[key][1] += value[1]
			else:
				result_dict.setdefault(key, [])
				result_dict[key].append([value[0]])
				result_dict[key].append([value[1]])

	result_array = []
	for result in result_dict.keys():
		result_array.append({
			'id': result,
			'circ': (result_dict[result][1][0] / len(result_dict[result][0])),
		})

	print('Circularity feature extraction finished. {} features extracted.'.format(len(result_array)))
	return (result_array)

### HELPER SECTION ###
def calc_props(idx, fns, prop_list=['size', 'slices', 'centroid', 'circ']):
	'''
	Helper function for 'compute_regions'
	:param fns: (string) list of filenames. sorted.
	:param prop_list: (list) of (strings) that contain the properties that should be stored in result.
	:returns result: (dict) with each segment. key: idx of segment -- value: [number of pixels in segment, idx of slice].
	'''
	result = {}
	if os.path.exists(fns):
		tmp = imageio.imread(fns)
		#labels, num_labels = np.unique(tmp, return_counts=True)
		regions = regionprops(tmp, cache=False)

		labels = []
		num_labels = []
		c_list = []
		circ_list = []
		for props in regions:
			labels.append(props.label)
			if any('size' in s for s in prop_list):
				num_labels.append(props.area)
			if any('centroid' in s for s in prop_list):
				c_list.append(tuple(map(int, props.centroid)))
			if any('circ' in s for s in prop_list):
				circ_list.append(cc(props.area, props.perimeter))

		for l in range(len(labels)):
			if labels[l] == 0:
				continue
			result.setdefault(labels[l], [])
			if any('size' in s for s in prop_list):
				result[labels[l]].append(num_labels[l])
			if any('slices' in s for s in prop_list):
				result[labels[l]].append(idx)
			if any('centroid' in s for s in prop_list):
				result[labels[l]].append(c_list[l])
			if any('circ' in s for s in prop_list):
				result[labels[l]].append(circ_list[l])

	return result

def cc(area, perimeter):
	'''
	The circularity of a circle is 1, and much less than one for a starfish footprint.
	'''
	if math.isnan(perimeter) or (perimeter == 0):
		circ = np.float64(0.0)
	else:
		circ = (4 * np.pi * area) / (perimeter**2)
	return (circ)

#### deprecated ####
def compute_intentsity(vol, gt, mode='3d'):
	'''
	This function takes both em and gt in order to compute the intensities from each segment.
	:param vol: volume (np.array) that contains the bare em data. (2d || 3d)
	:param gt: volume (np.array) that contains the groundtruth. (2d || 3d)

	:returns labels: (np.array) same shape as volume with all the labels.
	:returns intns: (np.array) is a vetor that contains the intensity values for every label.
	'''
	intns_values = []
	labels = np.zeros(shape=vol.shape, dtype=np.uint16)
	label_cnt = 0

	if mode == '2d':
		raise NotImplementedError('no 2d mode in this function yet.')
	else:
		if vol.ndim <= 2:
			raise ValueError('Volume is lacking on dimensionality(at least 3d): {}'.format(vol.shape))

		labels, num_label = label(gt, return_num=True)
		regions = regionprops(labels, cache=False)

		for props in regions:
			areat = props.area
			sumt = np.sum(vol[labels == props.label])
			intns_values.append(sumt / areat)

	intns = np.array(intns_values, dtype=np.float16)

	return (labels, intns)
