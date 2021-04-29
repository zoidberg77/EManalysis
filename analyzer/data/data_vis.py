import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import matplotlib.patches as patches
import open3d as o3d

import cv2

def visvol(vol, gt=None, add=None, filename='test', ff='png', save=False, dpi=500):
	'''
	Visualizing data and results for simplicity.
	:param vol: (np.array) em data volume.
	:param gt: (np.array) groundtruth data volume.
	:param add: (np.array) additional volume that can be shown.
 	'''
	if vol.ndim >= 3:
		raise ValueError('The input volume is higher than 2 dimensions.')
	else:
		fig = plt.figure()
		if gt is not None and add is not None:
			axes_list = list()
			axes_list.append(zero_to_nan(gt))
			axes_list.append(zero_to_nan(add))
			for i, element in enumerate(axes_list):
				fig.add_subplot(1, 2, i + 1)
				plt.axis('off')
				plt.imshow(vol, cmap = plt.cm.gray)
				plt.imshow(element, cmap='gist_ncar', alpha=0.8)
		else:
			plt.axis('off')
			plt.imshow(vol, cmap = plt.cm.gray)
			if gt is not None and add is None:
				gt = zero_to_nan(gt)
				#plt.imshow(gt, cmap='terrain', alpha=0.8)
				plt.imshow(gt, cmap='gist_ncar', alpha=0.8)
			if gt is None and add is not None:
				add = zero_to_nan(add)
				plt.imshow(add, cmap='gist_ncar', alpha=0.8)

		if save == True:
			fn = filename + '.' + ff
			plt.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches = 0)
		plt.close()

def visptc(ptc):
	'''
	Visualize the pointclouds
	'''
	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(ptc)
	point_cloud.paint_uniform_color([0.9, 0.1, 0.1])
	o3d.visualization.draw_geometries([point_cloud])

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

	points = [[bbox[2],bbox[0]],[bbox[2],bbox[1]],[bbox[3],bbox[0]],[bbox[3],bbox[1]]]
	plt.plot(*zip(*points), marker='o', color='r', ls='')
	ax.imshow(image, cmap="gray")
	plt.show()

### HELPER SECTION ###
def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	values = values.astype('float32')
	values[ values==0 ] = np.nan
	return values
