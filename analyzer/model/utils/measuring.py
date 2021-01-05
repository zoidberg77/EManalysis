import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

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

	if mode == '2d':
		if vol.ndim >= 3:
			for idx in range(vol.shape[0]):
				image = vol[idx, :, :]
				label, num_label = label(image, return_num=True)
				regions = regionprops(label)

				if idx != 0:
					label[label != 0] += num_label
					labels[idx] = label
				else:
					labels[idx] = label

				for props in regions:
					area_values.append(props.area)

	areas = np.array(area_values, dtype=np.uint16)

	plot_stats(size_values, 'number of labels', 'areasize')

	return (labels, areas)

def recompute_from_res(labels, result):
	'''
	Take the result labels from clustering algorithm and adjust the old labels.
	:param labels: (np.array) old labels just want to adjust.
	:param result: (np.array)
	'''
	pass

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
