import numpy as np
import glob
from analyzer.data.data_raw import readvol


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
				 sample_volume_size=(8, 64, 64),
				 sample_label_size=(8, 64, 64),
				 mode='2d',
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

		self.sample_volume_size = sample_volume_size
		self.sample_label_size = sample_label_size

		self.mode = mode
		self.file_format = file_format

	def load_chunk(self):
		'''
		Load chunk of em and groundtruth data for further processing.
		'''
		emfns = sorted(glob.glob(self.volpath + '*.' + self.file_format))
		gtfns = sorted(glob.glob(self.gtpath + '*.' + self.file_format))

		if self.mode == '2d':
			emdata = readvol(emfns[0])
			emdata = emdata[0, :, :]
			gt = readvol(gtfns[0])
			gt = gt[0, :, :]


		return (emdata, gt)
