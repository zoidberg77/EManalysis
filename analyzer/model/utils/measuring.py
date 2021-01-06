import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from analyzer.data.data_vis import visvol

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
		if vol.ndim >= 3:
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

	areas = np.array(area_values, dtype=np.uint16)

	#plot_stats(area_values, 'number of labels', 'areasize')

	return (labels, areas)

def recompute_from_res(labels, result):
	'''
	Take the result labels from clustering algorithm and adjust the old labels.
	:param labels: (np.array) old labels just want to adjust.
	:param result: (np.array)
	'''
	new = np.zeros(shape=labels.shape)

	for r in range(labels.shape[0]):
		tmp = labels[r]
		for idx in range(np.amin(tmp[np.nonzero(tmp)]), np.amax(tmp) + 1):
			tmp[tmp == idx] = result[idx - 1] + 1 # + 1 in order to secure that label 0 is not missed.

		new[r] = tmp

	return new

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
