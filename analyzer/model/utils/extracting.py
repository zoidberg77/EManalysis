import json

import numpy as np
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

	if dprc == 'iter':
		with multiprocessing.Pool(processes=kernel_n) as pool:
			tmp = pool.starmap(self.calc_props, enumerate(fns))

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

	return (result_array)


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


def compute_dist_graph(vol, dprc='full', fns=None, mode='3d'):
	'''
	This function computes a graph matrix that represents the distances from each segment to all others.
	:param vol: volume (np.array) that contains the groundtruth mask (= labels). (2d || 3d)
	:param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
	:returns: (np.array) (N x N) matrix gives you the feature vector--> N: number of segments
	'''
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

	if dprc == 'iter':
		raise NotImplementedError('no iter option yet.')

		#if
		#with multiprocessing.Pool(processes=cpus) as pool:
			#print("test")

	return (result_array)


### HELPER SECTION ###
def calc_props(idx, fns):
	'''
	Helper function for 'compute_regions'
	:param fns: (string) list of filenames. sorted.
	:returns result: (dict) with each segment. key: idx of segment -- value: [number of pixels in segment, idx of slice].
	'''
	result = {}
	if os.path.exists(fns):
		tmp = imageio.imread(fns)
		labels, num_labels = np.unique(tmp, return_counts=True)

		for l in range(labels.shape[0]):
			if labels[l] == 0:
				continue
			result.setdefault(labels[l], [])
			result[labels[l]].append(num_labels[l])

	return result

def plot_stats(data, x_label='x', y_label='y'):
	'''
	Plotting statistical data in order to get an impression.
	:param data: (np.array) potentially any dimension.
	:param x_label && y_label: (string) description of the x and y axis.
	'''
	if type(data).__name__ == 'list':
		plt.plot(data)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.show()
