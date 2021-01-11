import numpy as np
import glob
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

	def list_segments(self, vol, label, mode='2d'):
		'''
		This function creats a list of arrays that contain the unique segments.
		:param vol: (np.array) volume that contains the bare em data. (2d || 3d)
		:param label: (np.array) volume that contains the groundtruth. (2d || 3d)
		:returns: list of (np.array) objects that contain the segments.
		'''
		if mode == '2d':
			mask = np.zeros(shape=vol.shape, dtype=np.uint16)
		    mask[label > 0] = 1
		    vol[mask == 0] = 0

			for idx in range(vol.shape[0]):
				image = vol[idx, :, :]
				label2d, num_label = label(image, return_num=True)
				regions = regionprops(label2d, cache=False)

				for props in regions:
					boundbox = props.bbox


		elif mode == '3d':
			raise NotImplementedError('no 3d mode in this function yet.')
		else:
			raise ValueError('No valid dimensionality mode in function list_segments.')
