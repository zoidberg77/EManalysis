import numpy as np
from matplotlib import pyplot as plt

def visvol(vol, gt=None):
	'''
	Visualizing data and results for simplicity.
	:param vol: (np.array) em data volume.
	:param vol: (np.array) groundtruth data volume.
	'''
	if vol.ndim >= 3:
		raise ValueError('The input volume is higher than 2 dimensions.')
	else:
		plt.figure()
		plt.imshow(vol, cmap = plt.cm.gray)
		if gt is not None:
			gt = zero_to_nan(gt)
			#plt.imshow(gt, cmap='terrain', alpha=0.9)
			plt.imshow(gt, cmap='gist_ncar', alpha=0.1)
		plt.show()


### HELPER SECTION ###

def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	values[ values==0 ] = np.nan
	return values
