import numpy as np
from matplotlib import pyplot as plt

def visvol(vol, gt=None):
	'''
	Visualizing data and results for simplicity.
	:param vol: (np.array) em data volume.
	:param vol: (np.array) groundtruth data volume.
	'''
	plt.figure()
	plt.imshow(vol, cmap = plt.cm.gray)
	if gt is not None:
		plt.imshow(gt, cmap='Reds', alpha=0.4)
	plt.show()
