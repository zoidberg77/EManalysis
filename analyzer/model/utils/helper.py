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
