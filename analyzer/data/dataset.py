import numpy as np
import glob
from skimage.measure import label, regionprops
from analyzer.data.data_vis import visvol
from analyzer.data.data_raw import readvol, folder2Vol


class Dataloader():
	'''
	Dataloader class for handling the em dataset and the related labels.

	:param volpath: (string) path to the directory that contains the em volume(s).
	:param gtpath: (string) path to the directory that contains the groundtruth volume(s).
	:param volume:
	:param label:
	:param chunk_size: (tuple) defines the chunks in which the data is loaded. Can help to overcome Memory errors.
	:param mode: (string) Sets the mode in which the volume should be clustered (3d || 2d).
	:param ff: (string) defines the file format that you want to work with. (default: png)
	'''
	def __init__(self,
				 volpath,
				 gtpath,
				 volume=None,
				 label=None,
				 chunk_size=(100,4096,4096),
				 mode='3d', ff='png'
				 ):

		if volume is not None:
			pass
		else:
			self.volpath = volpath
			self.volume = volume

		if label is not None:
			pass
		else:
			self.gtpath = gtpath
			self.label = label

		self.chunk_size = chunk_size
		self.mode = mode
		self.ff = ff


	def load_chunk(self, vol='both'):
		'''
		Load chunk of em and groundtruth data for further processing.
		:param vol: (string) choose between -> 'both', 'em', 'gt' in order to specify
					 with volume you want to load.
		'''
		emfns = sorted(glob.glob(self.volpath + '*.' + self.ff))
		gtfns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
		emdata = 0
		gt = 0

		if self.mode == '2d':
			if (vol == 'em') or (vol == 'both'):
				emdata = readvol(emfns[0])
				emdata = np.squeeze(emdata)
				print('em data loaded: ', emdata.shape)
			if (vol == 'gt') or (vol == 'both'):
				gt = readvol(gtfns[0])
				gt = np.squeeze(gt)
				print('gt data loaded: ', gt.shape)

		if self.mode == '3d':
			if (vol == 'em') or (vol == 'both'):
				if self.volume is None:
					emdata = folder2Vol(self.volpath, self.chunk_size, file_format=self.ff)
					print('em data loaded: ', emdata.shape)
			if (vol == 'gt') or (vol == 'both'):
				if self.label is None:
					gt = folder2Vol(self.gtpath, self.chunk_size, file_format=self.ff)
					print('gt data loaded: ', gt.shape)

		return (emdata, gt)

	def list_segments(self, vol, labels, min_size=2000, os=0, mode='2d'):
		'''
		This function creats a list of arrays that contain the unique segments.
		:param vol: (np.array) volume that contains the bare em data. (2d || 3d)
		:param label: (np.array) volume that contains the groundtruth. (2d || 3d)
		:param min_size: (int) this sets the minimum size of mitochondria region in order to be safed to the list. Used only in 2d.
		:param os: (int) defines the offset that should be used for cutting the bounding box. Be careful with offset as it can lead to additional regions in the chunks.
		:param mode: (string) 2d || 3d --> 2d gives you 2d arrays of each slice (same mitochondria are treated differently as they loose their touch after slicing)
									   --> 3d gives you the whole mitochondria in a 3d volume.

		:returns: (list) of (np.array) objects that contain the segments.
		'''
		bbox_list = []

		mask = np.zeros(shape=vol.shape, dtype=np.uint16)
		mask[labels > 0] = 1
		vol[mask == 0] = 0

		if mode == '2d':
			for idx in range(vol.shape[0]):
				image = vol[idx, :, :]
				gt_img = labels[idx, :, :]
				label2d, num_label = label(gt_img, return_num=True)
				regions = regionprops(label2d, cache=False)

				for props in regions:
					boundbox = props.bbox
					if props.bbox_area > min_size:
						if ((boundbox[0] - os) < 0) or ((boundbox[2] + os) > image.shape[0]) or ((boundbox[1] - os) < 0) or ((boundbox[3] + os) > image.shape[1]):
							tmparr = image[boundbox[0]:boundbox[2], boundbox[1]:boundbox[3]]
						else:
							tmparr = image[(boundbox[0] - os):(boundbox[2] + os), (boundbox[1] - os):(boundbox[3] + os)]

						bbox_list.append(tmparr)

		elif mode == '3d':
			chunk_dict = {}

			label3d, num_label = label(labels, return_num=True)
			regions = regionprops(label3d, cache=False)

			for props in regions:
				boundbox = props.bbox

				if ((boundbox[1] - os) < 0) or ((boundbox[4] + os) > vol.shape[1]) or ((boundbox[2] - os) < 0) or ((boundbox[5] + os) > vol.shape[2]):
					tmparr = vol[boundbox[0]:boundbox[3], boundbox[1]:boundbox[4], boundbox[2]:boundbox[5]]
				else:
					tmparr = vol[boundbox[0]:boundbox[3], (boundbox[1] - os):(boundbox[4] + os), (boundbox[2] - os):(boundbox[5] + os)]

				bbox_list.append(tmparr)

		else:
			raise ValueError('No valid dimensionality mode in function list_segments.')

		return (bbox_list)
