import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from analyzer.data.data_vis import visvol

import time


def compute_regions(vol, mode='2d'):
	'''
	Compute the region properties of the groundtruth labels.

	:param vol: volume (np.array) that contains the labels. (2d || 3d)
	:param mode: (string)

	:returns labels: (np.array) same shape as volume with all the labels.
	:returns areas: (np.array) is a vetor that contains a area for every label.
	'''
	area_values = []
	labels = np.zeros(shape=vol.shape, dtype=np.uint16)
	label_cnt = 0

	if mode == '2d':
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

		labels, num_label = label(vol, return_num=True)
		regions = regionprops(labels, cache=False)

		for props in regions:
			area_values.append(props.area)

	areas = np.array(area_values, dtype=np.uint16)

	#plot_stats(area_values, 'number of labels', 'areasize')

	return (labels, areas)

def recompute_from_res(labels, result, mode='2d'):
	'''
	Take the result labels from clustering algorithm and adjust the old labels. NOTE: '3d' mode is way faster.
	:param labels: (np.array) old labels just want to adjust.
	:param result: (np.array)
	'''
	tic = time.perf_counter()

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

	toc = time.perf_counter()
	#print(f"function needed {toc - tic:0.4f} seconds")

	return cld_labels


### HELPER SECTION ###
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
