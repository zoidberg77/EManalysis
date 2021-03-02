import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import matplotlib.patches as patches

import cv2

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
			plt.imshow(gt, cmap='gist_ncar', alpha=0.8)
		plt.show()

def vissegments(image, segments, mask=None):
	'''
	Visualize volume and the overlaying segments created by a superpixels algorithm.
	:param image: (np.array) em data volume slice.
	:param segments: (np.array)
	'''
	fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
	ax1, ax2, ax3, ax4 = ax_arr.ravel()
	ax1.imshow(image, cmap="gray")
	ax1.set_title("EM data image")

	if mask is not None:
		ax2.imshow(mask, cmap="gray")
		ax2.set_title("Mask")

	ax3.imshow(image, cmap="gray")
	ax3.imshow(mark_boundaries(image, segments), alpha=0.4)
	ax3.set_title("SLIC segments")

	plt.show()

def visbbox(image, bbox):
	'''
	Draw bounding box in order to check if it is correct.
	:param image: (np.array) data volume slice.
	:param bbox: (tuple) rmin, rmax, cmin, cmax
	'''
	fig,ax = plt.subplots()

	rect = patches.Rectangle((bbox[1],bbox[2]), (bbox[3] - bbox[2]), (bbox[1] - bbox[0]), linewidth=1, edgecolor='r', facecolor='none')

	#print(bbox)
	points = [[bbox[2],bbox[0]],[bbox[2],bbox[1]],[bbox[3],bbox[0]],[bbox[3],bbox[1]]]
	print(points)
	plt.plot(*zip(*points), marker='o', color='r', ls='')
	ax.imshow(image, cmap="gray")
	#ax.add_patch(rect)

	plt.show()


### HELPER SECTION ###
def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	values = values.astype(np.float16)
	values[ values==0 ] = np.nan
	return values
