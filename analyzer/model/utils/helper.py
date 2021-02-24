import numpy as np
from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix

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

def recompute_from_res(vol, labels, result, dprc='full', mode='3d'):
	'''
	Take the result labels from clustering algorithm and adjust the old labels. NOTE: '3d' mode is way faster.
	:param vol: (np.array) matrix that is the groundtruth mask.
	:param labels: (np.array) vector that contains old labels that you want to adjust.
	:param result: (np.array) vector that contains the new labels.
	:param dprc: (string)
	:returns cld_labels: (np.array) vol matrix that is the same shape as vol mask. But with adjusted labels.
	'''
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
		#TODO!
		raise NotImplementedError('no iterative option in this function yet.')
	else:
		raise ValueError('No valid data processing option choosen. Please choose \'full\' or \'iter\'.')

	return cld_labels

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
