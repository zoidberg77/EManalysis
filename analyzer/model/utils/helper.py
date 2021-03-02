import os, sys
import numpy as np
import multiprocessing
import functools
import imageio
from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix

from analyzer.data.data_raw import save_m_to_image

def convert_to_sparse(inputs):
	'''
	Convert any sort of input (list of different sized arrays) to a sparse matrix
	as preprocessing step in order to cluster later. Sparse has added 0 in every feature vector.
	:param input: (list) or (dict) of feature vectors that differ in shape.

	:returns sparse: (M, N) matrix. Rows represent segment respectively to the res_labels (one full feature vector).
									columns represent the individual features.
	'''
	if type(inputs) is dict:
		in_list = list(inputs.values())
	elif type(inputs) is list:
		in_list = inputs
	else:
		raise ValueError('Input type is not supported for \'convert_to_sparse\' function.')

	in_list = list(inputs.values())
	row = len(inputs)
	column = int(max(arr.shape[0] for arr in in_list if arr.size != 0))

	sparse = np.zeros(shape=(row, column), dtype=np.float64)
	for idx in range(len(in_list)):
		tmp = in_list[idx]
		if in_list[idx].shape[0] < column:
			tmp = np.append(in_list[idx], np.zeros((column - in_list[idx].shape[0], )), axis=0)
		sparse[idx] = tmp

	return (sparse)

def recompute_from_res(labels, result, vol= None, volfns=None, dprc='full', fp='', mode='3d'):
	'''
	Take the result labels from clustering algorithm and adjust the old labels. NOTE: '3d' mode is way faster.
	:param labels: (np.array) vector that contains old labels that you want to adjust.
	:param result: (np.array) vector that contains the new labels.
	:param vol: (np.array) matrix that is the groundtruth mask.
	:param volfns: (list) of image filenames that contain the groundtruth mask.
	:param fp: (string) this should give you the folder path where the resulting image should be stored.
	:param dprc: (string)
	:returns cld_labels: (np.array) vol matrix that is the same shape as vol mask. But with adjusted labels.
	'''
	print('Starting to relabel the mitochondria.')
	if dprc == 'full':
		if mode == '2d':
			cld_labels = np.zeros(shape=labels.shape)

			for r in range(labels.shape[0]):
				tmp = labels[r]
				for idx in range(np.amin(tmp[np.nonzero(tmp)]), np.amax(tmp) + 1):
					tmp[tmp == idx] = result[idx - 1] + 1 # + 1 in order to secure that label 0 is not missed.

				cld_labels[r] = tmp
		else:
			ldict = {}
			for k, v in zip(labels, result):
				ldict[k] = v + 1  # + 1 in order to secure that label 0 is not missed.

			k = np.array(list(ldict.keys()))
			v = np.array(list(ldict.values()))

			mapv = np.zeros(k.max() + 1)
			mapv[k] = v
			cld_labels = mapv[vol]
	elif dprc == 'iter':
		ldict = {}
		for k, v in zip(labels, result):
			ldict[k] = v + 1  # + 1 in order to secure that label 0 is not missed.

		k = np.array(list(ldict.keys()))
		v = np.array(list(ldict.values()))

		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			pool.starmap(functools.partial(recompute_from_res_per_slice, k=k, v=v, fp=fp), enumerate(volfns))
		cld_labels = 0 #Just to avoid error message.
	else:
		raise ValueError('No valid data processing option choosen. Please choose \'full\' or \'iter\'.')
	print('Relabeling of the mitochondria is done.')
	return cld_labels

def recompute_from_res_per_slice(idx, fns, k, v, fp):
	'''
	Helper function to iterate over the whole dataset in order to replace the labels with its
	clustering labels.
	'''
	if os.path.exists(fns):
		vol = imageio.imread(fns)
		mapv = np.zeros(k.max() + 1)
		mapv[k] = v
		cld_labels = mapv[vol]
	else:
		raise ValueError('image {} not found.'.format(fns))
	save_m_to_image(cld_labels, 'cluster_mask', fp=fp, idx=idx, ff='png')

def convert_dict_mtx(inputs, valn):
	'''
	This function converts a dict with labels as keys and values to 2 separate matrices that represent
	feature vectors/matrix and labels vector.
	:param input: (dict) or (list)
	:param valn: (string) name of value parameter in dict.
	:returns labels: (np.array) same shape as volume with all the labels.
	:returns values: (np.array) is a vetor that contains the corresponding values for every label.
	'''
	if (type(inputs) is list):
		labels = np.array([seg['id'] for seg in inputs])
		if isinstance(inputs[0][valn], (list, tuple, np.ndarray)) is False:
			values = np.array([seg[valn] for seg in inputs])
		else:
			values = np.concatenate([seg[valn] for seg in inputs])
	elif (type(inputs) is dict):
		labels, values = zip(* inputs.items())
		labels = np.array(labels, dtype=np.uint16)
		values = np.array(values, dtype=np.uint16)
	else:
		raise TypeError('input type {} can not be processed.'.format(type(inputs)))

	return (labels, values)

def min_max_scale(X, desired_range=(0,1)):
	'''
	Transform features by scaling each feature to a given range.
	:param X: Matrix you want to transform.
	:param range: Desired range of transformed data.
	:returns X_scaled: scaled matrix.
	'''
	min_v = desired_range[0]
	max_v = desired_range[1]
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max_v - min_v) + min_v
	return X_scaled
