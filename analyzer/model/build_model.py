import os, sys
import numpy as np
import pandas as pd
from skimage.measure import label
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN
from analyzer.model.utils.extracting import *
from analyzer.model.utils.superpixel import superpixel_segment, superpixel_image, texture_analysis
from analyzer.model.utils.helper import convert_to_sparse, recompute_from_res, convert_dict_mtx
from analyzer.data.data_vis import visvol, vissegments
from analyzer.utils.eval import clusteravg

from .feat_extr_model import FeatureExtractor

class Clustermodel():
	'''
	Setups up the model for running a clustering algoritm on the loaded data.
	:param emvol & gtvol: (np.array) Both are the data volumes.
	:param dl: (class object) This is the dataloader class object.
	:param alg: sets the clustering algorithm that should be used. (default: KMeans)
				- 'kmeans': KMeans
				- 'affprop': AffinityPropagation
				- 'specCl': SpectralClustering
				- 'dbscan': DBSCAN

	:param clstby: choose how you want to cluster and label the segments.
				- 'bysize': cluster segements by their size.
				- 'bytext': cluster segments by texture. EM needed.

	:param n_cluster: (int) sets the number of cluster that should be found.
	:param mode: (string) Analyze either by 2d or 3d slizes.
	'''
	def __init__(self, cfg, emvol, gtvol, dl=None, clstby='bysize', n_cluster=5):
		self.cfg = cfg
		self.emvol = emvol
		self.gtvol = gtvol
		self.dl = dl
		self.alg = self.cfg.CLUSTER.ALG
		self.clstby = clstby
		self.feat_list = self.cfg.CLUSTER.FEAT_LIST
		self.n_cluster = n_cluster
		self.mode = self.cfg.MODE.DIM

		self.model = self.set_model(mn=self.alg)
		self.fe = FeatureExtractor(self.cfg)

		print(' --- model is set. algorithm: {}, clustering: {} , features: {} --- '.format(self.alg, self.clstby, str(self.feat_list).strip('[]')))

	def set_model(self, mn='kmeans'):
		'''
		This function enables the usage of different algoritms when setting the model overall.
		:param mn: (string) that is the name of the algoritm to go with.
		'''
		if mn == 'kmeans':
			model = KMeans(n_clusters=self.n_cluster)
		elif mn == 'affprop':
			model = AffinityPropagation()
		elif mn == 'specCl':
			model = SpectralClustering(n_clusters=self.n_cluster)
		elif mn == 'dbscan':
			model = DBSCAN()
		else:
			raise ValueError('Please enter a valid clustering algorithm. -- \'kmeans\', \'affprop\', \'specCl\', \'dbscan\'')

		return model

	def load_features(self, feature_list=['sizef', 'distf', 'vaef']):
		'''
		This function will load different features vectors that were extracted and saved to be used for clustering.
		'''
		rs_feat_list = list()
		for fns in feature_list:
			if os.path.exists(self.cfg.DATASET.ROOTF + fns + '.json') is False:
				print('This file {} does not exist, will be computed.'.format(self.cfg.DATASET.ROOTF + fns + '.json'))

				if fns == 'sizef':
					feat = self.fe.compute_seg_size()
				elif fns == 'distf':
					feat = self.fe.compute_seg_dist()
				elif fns == 'vaef':
					feat = self.fe.infer_vae()
				else:
					print('No function for computing {} features.'.format(fns))
			else:
				fn = self.cfg.DATASET.ROOTF + fns + '.json'
				with open(fn, 'r') as f:
					feat = json.loads(f.read())
				rs_feat_list.append(feat)

		return rs_feat_list

	def stack_features(self, feature_list=['sizef', 'distf', 'vaef']):
		'''
		This function takes different features and stacks them togehter for further clustering.
		'''
		#rs_feat_list = self.load_features(feature_list=feature_list)
		raise NotImplementedError

	def run(self):
		if self.clstby == 'bysize':
			# RUN the clustering by size parameters.
			test = self.load_features(feature_list=['sizef'])[0]
			print(len(test))
			rst = compute_region_size(self.gtvol, mode=self.mode)
			print(len(rst))
			labels, areas = convert_dict_mtx(rst, 'size')
			res_labels = self.model.fit_predict(np.array(areas).reshape(-1,1))
			#res_labels = self.model.fit_predict(pd.DataFrame(rst))

			labeled = recompute_from_res(self.gtvol, labels, res_labels, mode=self.mode)
			for k in range(labeled.shape[0]):
				visvol(self.emvol[k], labeled[k])

		elif self.clstby == 'bytext':
			# RUN the clustering by texture parameters.
			volume = self.emvol.copy()
			if self.dl is not None:
				labels = label(self.gtvol)
				segments = self.dl.list_segments(self.emvol, self.gtvol, mode=self.mode)

				texture_dict = texture_analysis(segments)
				sparse = convert_to_sparse(texture_dict)

				res_labels = self.model.fit_predict(sparse)
				labeled = recompute_from_res(labels, res_labels, mode=self.mode)

				for k in range(self.emvol.shape[0]):
					visvol(volume[k], labeled[k])
			else:
				raise ValueError('No dataloader functionality useable as (dl == None).')

		elif self.clstby == 'bydist':
			# RUN the clustering by distance graph parameters.
			rst = compute_dist_graph(self.gtvol, mode=self.mode)
			labels = [seg['id'] for seg in rst]
			dist_m = np.vstack([seg['dist'] for seg in rst])

			res_labels = self.model.fit_predict(dist_m)
			labeled = recompute_from_res(self.gtvol, labels, res_labels, mode=self.mode)

			for k in range(self.emvol.shape[0]):
				visvol(self.emvol[k], labeled[k])

		elif self.clstby == 'byall':
			#RUN the clustering by using all the features extracted.
			pass

		else:
			raise Exception('Please state according to which property should be clustered.')
