import os, sys
import numpy as np
import json
import glob
import h5py
from numpyencoder import NumpyEncoder

from analyzer.model.utils.extracting import compute_region_size, compute_dist_graph, compute_circularity, compute_intentsity
from analyzer.model.utils.helper import convert_dict_mtx

class FeatureExtractor():
	'''
	Using this model to build up your feature matrix that will be clustered.
	:param emvol & gtvol: (np.array) Both are the data volumes.
	:param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
				This is important as you might face memory problems loading the whole dataset into your RAM. Distinguish between two setups:
				- 'full': This enables reading the whole stack at once. Or at least the 'chunk_size' you set.
				- 'iter': This iterates over each slice/image and extracts information one by one.
						  This might help you to process the whole dataset without running into memory error.
	'''
	def __init__(self, cfg, emvol=None, gtvol=None):
		self.cfg = cfg
		self.emvol = emvol
		self.gtvol = gtvol
		self.empath = self.cfg.DATASET.EM_PATH
		self.gtpath = self.cfg.DATASET.LABEL_PATH
		self.dprc = self.cfg.MODE.DPRC
		self.ff = self.cfg.DATASET.FILE_FORMAT
		self.mode = self.cfg.MODE.DIM

		if self.dprc == 'iter':
			self.emfns = sorted(glob.glob(self.empath + '*.' + self.ff))
			self.gtfns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
		else:
			self.emfns = None
			self.gtfns = None

	def compute_seg_size(self):
		'''
		Extract the size of each mitochondria segment.
		:returns result_dict: (dict) where the label is the key and the size of the segment is the corresponding value.
		'''
		return compute_region_size(self.gtvol, fns=self.gtfns, dprc=self.dprc, mode=self.mode)

	def compute_seg_dist(self):
		'''
		Compute the distances of mitochondria to each other and extract it as a graph matrix.
		:returns
		'''
		return compute_dist_graph(self.gtvol, fns=self.gtfns, dprc=self.dprc, mode=self.mode)

	def infer_vae(self):
		'''
		Function runs the vae option.
		'''
		print('Not there yet for vae features.')
		#raise NotImplementedError

	def compute_seg_circ(self):
		'''
		Computes the circularity features from mitochondria volume.
		'''
		return compute_circularity(self.gtvol, fns=self.gtfns, dprc=self.dprc, mode=self.mode)

	def save_feat_h5(self, rsl_dict, filen='feature_vector'):
		'''
		Saving arrays to h5 that contains the features.
		:param rsl_dict: (dict) that contains features.
		:param filen: (string) filename.
		'''
		labels, values = convert_dict_mtx(rsl_dict, filen[:-1])
		with h5py.File(self.cfg.DATASET.ROOTF + filen + '.h5', 'w') as h5f:
			h5f.create_dataset('id', data=labels)
			h5f.create_dataset(filen[:-1], data=values)
			h5f.close()
		#with h5py.File(self.cfg.DATASET.ROOTF + filen + '.h5', 'w') as h5f:
			#ds = h5f.create_dataset(filen, shape=(len(labels)))
			#idx = h5f.create_group('id')
			#feat = h5f.create_group(filen[:-1])
			#idx = labels
			#feat = values
			#h5f.close()

	def save_feat_dict(self, rsl_dict, filen='feature_vector.json'):
		'''
		Saving dict that contains the features to the designated folder.
		:param rsl_dict: (dict) that contains features.
		:param filen: (string) filename.
		'''
		with open(os.path.join(self.cfg.DATASET.ROOTF + filen), 'w') as f:
			json.dump(rsl_dict, f, cls=NumpyEncoder)
			f.close()
