import numpy as np
import glob
from analyzer.data.data_raw import readvol, folder2Vol


class Dataloader():
	'''
	Dataloader class for handling the em dataset.

	:param
	'''
	def __init__(self,
				 volpath,
				 gtpath,
				 volume=None,
				 label=None,
				 chunk_size=(100,4096,4096),
				 sample_volume_size=(8, 64, 64),
				 sample_label_size=(8, 64, 64),
				 mode='3d',
				 file_format='png'
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
		self.sample_volume_size = sample_volume_size
		self.sample_label_size = sample_label_size

		self.mode = mode
		self.file_format = file_format

	def load_chunk(self, vol='both'):
		'''
		Load chunk of em and groundtruth data for further processing.
		:param vol: (string) choose between -> 'both', 'em', 'gt' in order to specify
					 with volume you want to load.
		'''
		emfns = sorted(glob.glob(self.volpath + '*.' + self.file_format))
		gtfns = sorted(glob.glob(self.gtpath + '*.' + self.file_format))
		emdata = 0
		gt = 0

		if self.mode == '2d':
			if (vol == 'em') or (vol == 'both'):
				emdata = readvol(emfns[0])
				emdata = np.squeeze(emdata)
			if (vol == 'gt') or (vol == 'both'):
				gt = readvol(gtfns[0])
				gt = np.squeeze(gt)

		if self.mode == '3d':
			if (vol == 'em') or (vol == 'both'):
				if self.volume is None:
					emdata = folder2Vol(self.volpath, self.chunk_size)
			if (vol == 'gt') or (vol == 'both'):
				if self.label is None:
					gt = folder2Vol(self.gtpath, self.chunk_size)

		return (emdata, gt)
