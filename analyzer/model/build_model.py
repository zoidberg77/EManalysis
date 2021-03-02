import os, sys
import numpy as np
import pandas as pd
import h5py
from skimage.measure import label
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analyzer.model.utils.extracting import *
from analyzer.model.utils.superpixel import superpixel_segment, superpixel_image, texture_analysis
from analyzer.model.utils.helper import convert_to_sparse, recompute_from_res, convert_dict_mtx, min_max_scale
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
	def __init__(self, cfg, emvol, gtvol, dl=None, clstby='bysize'):
		self.cfg = cfg
		self.emvol = emvol
		self.gtvol = gtvol
		self.dl = dl
		self.alg = self.cfg.CLUSTER.ALG
		self.clstby = clstby
		self.feat_list = self.cfg.CLUSTER.FEAT_LIST
		self.weightsf = self.cfg.CLUSTER.WEIGHTSF
		self.n_cluster = self.cfg.CLUSTER.N_CLUSTER
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

	def get_features(self, feature_list=['sizef', 'distf', 'vaef', 'circf']):
		'''
		This function will load different features vectors that were extracted and saved to be used for clustering.
		:param feat_list: (list) of (string)s that states which features should be computed and/or load to cache for
							further processing.
		:returns labels: (np.array) that contains the labels.
		:returns rs_feat_list: (list) of (np.array)s that contain the related features.
		'''
		rs_feat_list = list()
		labels = np.array([])
		for fns in self.feat_list:
			if os.path.exists(self.cfg.DATASET.ROOTF + fns + '.h5') is False:
				print('This file {} does not exist, will be computed.'.format(self.cfg.DATASET.ROOTF + fns + '.h5'))

				if fns == 'sizef':
					feat = self.fe.compute_seg_size()
				elif fns == 'distf':
					feat = self.fe.compute_seg_dist()
				elif fns == 'vaef':
					feat = self.fe.infer_vae()
				elif fns == 'circf':
					feat = self.fe.compute_seg_circ()
				else:
					print('No function for computing {} features.'.format(fns))
				#self.fe.save_feat_dict(feat, filen=(fns + '.json'))
				label, values = self.fe.save_single_feat_h5(feat, filen=fns)
				if labels.size == 0:
					labels = np.array(label)
				rs_feat_list.append(np.array(values))
			else:
				#fn = self.cfg.DATASET.ROOTF + fns + '.json'
				#with open(fn, 'r') as f:
					#feat = json.loads(f.read())
				fn = self.cfg.DATASET.ROOTF + fns + '.h5'
				with h5py.File(fn, "r") as h5f:
					if labels.size == 0:
						labels = np.array(h5f['id'])
					rs_feat_list.append(np.array(h5f[fns[:-1]]))

		return labels, rs_feat_list

	def stack_and_std_features(self, feat):
		'''
		This function takes different features and stacks them togehter and performs
		standardization by centering and scaling for further clustering.
		:param feat: (list) of features.
		'''
		scaler = StandardScaler()
		if len(feat) == 1:
			return feat

		feats = np.column_stack([singlef for singlef in feat])
		std_feat = scaler.fit_transform(feats)

		return std_feat

	def prep_cluster_matrix(self, labels, feat_list):
		'''
		Function computes clustering matrix from different features for the actual clustering.
		:param labels:
		:param feat_list: (list) of (np.array)s that are the feature vectors/matrices.
		:param weights: (np.array) of weighting factor for different features.
		:returns clst_m: (np.array) of NxN clustering distance from each feature to another. N is a sample.
		'''
		#Preload if possible.
		if os.path.exists(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')) \
                and os.stat(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')).st_size != 0:
			print('preload the clustering matrix.')
			with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5'), "r") as h5f:
				clst_m = np.array(h5f['clstm'])
				h5f.close()
		else:
			scaler = MinMaxScaler()
			clst_m = np.zeros(shape=feat_list[0].shape[0], dtype=np.float16)
			for idx, feat in enumerate(feat_list):
				if feat.ndim <= 1:
					tmp = scaler.fit_transform(feat.reshape(-1,1))
					#clst_m = clst_m + weights[idx] * distance.cdist(tmp, tmp, 'euclidean')
					clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * distance.cdist(tmp, tmp, 'euclidean'))
				else:
					#clst_m = clst_m + weights[idx] * min_max_scale(feat)
					clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * min_max_scale(feat))

			clst_m = np.vstack(clst_m)
			self.fe.save_feats_h5(labels, clst_m, filen='clstm')
		return clst_m

	def run(self):
		'''
		Running the main clustering algoritm on the features (feature list) extracted.
		'''
		labels, feat = self.get_features(feature_list=self.feat_list)
		clst_m = self.prep_cluster_matrix(labels, feat)
		print(clst_m.shape)
		res_labels = self.model.fit_predict(clst_m)
		print(res_labels)
		_, gtfns = self.fe.get_fns()
		recompute_from_res(labels, res_labels, volfns=gtfns, dprc=self.cfg.MODE.DPRC, fp=self.cfg.CLUSTER.OUTPUTPATH)

	'''
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
			labels, feat = self.load_features(feature_list=self.feat_list)

		else:
			raise Exception('Please state according to which property should be clustered.')
		'''
