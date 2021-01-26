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


def recompute_from_res(labels, result, mode='3d'):
	'''
	Take the result labels from clustering algorithm and adjust the old labels. NOTE: '3d' mode is way faster.
	:param labels: (np.array) old labels just want to adjust.
	:param result: (np.array)
	'''
	if mode == '2d':
		cld_labels = np.zeros(shape=labels.shape)

		for r in range(labels.shape[0]):
			tmp = labels[r]
			for idx in range(np.amin(tmp[np.nonzero(tmp)]), np.amax(tmp) + 1):
				tmp[tmp == idx] = result[idx - 1] + 1 # + 1 in order to secure that label 0 is not missed.

			cld_labels[r] = tmp
	else:
		tmp = np.arange(start=np.amin(labels[np.nonzero(labels)]), stop=np.amax(labels) + 1, step=1)
		ldict = {}
		for k, v in zip(tmp, result):
			ldict[k] = v + 1  # + 1 in order to secure that label 0 is not missed.

		k = np.array(list(ldict.keys()))
		v = np.array(list(ldict.values()))

		mapv = np.zeros(k.max() + 1)
		mapv[k] = v
		cld_labels = mapv[labels]

	return cld_labels
