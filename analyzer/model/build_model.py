import os, sys
import numpy as np
import h5py
import imageio
#import hdbscan
from scipy.spatial import distance
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
				- 'hdbscan': HDBSCAN (https://hdbscan.readthedocs.io/en/latest/index.html)

	:param clstby: choose how you want to cluster and label the segments.
				- 'bysize': cluster segements by their size.
				- 'bytext': cluster segments by texture. EM needed.

	:param n_cluster: (int) sets the number of cluster that should be found.
	:param mode: (string) Analyze either by 2d or 3d slizes.
	'''
	def __init__(self, cfg, emvol=None, gtvol=None, dl=None):
		self.cfg = cfg
		self.emvol = emvol
		self.gtvol = gtvol
		self.dl = dl
		self.alg = self.cfg.CLUSTER.ALG
		self.feat_list = self.cfg.CLUSTER.FEAT_LIST
		self.weightsf = self.cfg.CLUSTER.WEIGHTSF
		self.n_cluster = self.cfg.CLUSTER.N_CLUSTER

		self.model = self.set_model(mn=self.alg)
		self.fe = FeatureExtractor(self.cfg)

		print(' --- model is set. algorithm: {}, clustering by the features: {} --- '.format(self.alg, str(self.feat_list).strip('[]')))

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
		elif mn == 'hdbscan':
			#model = hdbscan.HDBSCAN(min_cluster_size=self.n_cluster, gen_min_span_tree=True)
			pass
		else:
			raise ValueError('Please enter a valid clustering algorithm. -- \'kmeans\', \'affprop\', \'specCl\', \'dbscan\', \'hdbscan\'')

		return model

	def get_features(self):
		'''
		This function will load different features vectors that were extracted and saved to be used for clustering.
		:param feat_list: (list) of (string)s that states which features should be computed and/or load to cache for
							further processing.
		:returns labels: (np.array) that contains the labels.
		:returns rs_feat_list: (list) of (np.array)s that contain the related features.
		'''
		rs_feat_list = list()
		labels = np.array([])
		for idx, fns in enumerate(self.feat_list):
			if os.path.exists(self.cfg.DATASET.ROOTF + fns + '.h5') is False:
				print('This file {} does not exist, will be computed.'.format(self.cfg.DATASET.ROOTF + fns + '.h5'))

				if fns == 'sizef':
					feat = self.fe.compute_seg_size()
				elif fns == 'distf':
					feat = self.fe.compute_seg_dist()
				elif fns == 'shapef':
					feat = self.fe.compute_vae_shape()
				elif fns == 'textf':
					feat = self.fe.compute_vae_texture()
				elif fns == 'circf':
					feat = self.fe.compute_seg_circ()
				else:
					print('No function for computing {} features.'.format(fns))

				label, values = self.fe.save_single_feat_h5(feat, filen=fns)
				if labels.size == 0:
					labels = np.array(label)
				rs_feat_list.append(np.array(values))
			else:
				fn = self.cfg.DATASET.ROOTF + fns + '.h5'
				with h5py.File(fn, "r") as h5f:
					if labels.size == 0:
						labels = np.array(h5f['id'])

					test = np.array(h5f[fns[:-1]])
					print('Features {} have shape {}'.format(fn, test.shape))
					rs_feat_list.append(np.array(h5f[fns[:-1]]))
					print('Loaded {} features to cache.'.format(fns[:-1]))

			if idx == 0:
				pass
				#labels = base_labels
			else:
				#correct_idx_feat(
				pass

		return labels, rs_feat_list

	def prep_cluster_matrix(self, labels, feat_list, load=False, save=False):
		'''
		Function computes clustering matrix from different features for the actual clustering.
		:param labels:
		:param feat_list: (list) of (np.array)s that are the feature vectors/matrices.
		:param weights: (np.array) of weighting factor for different features.
		:returns clst_m: (np.array) of NxN clustering distance from each feature to another. N is a sample.
		'''
		#Preload if possible.
		if load and os.path.exists(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')) \
                and os.stat(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')).st_size != 0:
			print('preload the clustering matrix.')
			with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5'), "r") as h5f:
				clst_m = np.array(h5f['clstm'])
				h5f.close()
		else:
			print('computing the clustering matrix.')
			scaler = MinMaxScaler()
			clst_m = np.zeros(shape=feat_list[0].shape[0], dtype=np.float16)
			for idx, feat in enumerate(feat_list):
				if feat.shape[0] == feat.shape[1]:
					clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * min_max_scale(feat))
				elif feat.ndim <= 1:
					tmp = scaler.fit_transform(feat.reshape(-1,1))
					clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * distance.cdist(tmp, tmp, 'euclidean'))
				else:
					tmp = min_max_scale(feat)
					clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * distance.cdist(tmp, tmp, 'euclidean'))

			clst_m = np.vstack(clst_m)
			if save == True:
				self.fe.save_feats_h5(labels, clst_m, filen='clstm')
		return clst_m

	def visualize(self, end=True):
		'''
		Visualize some results.
		'''
		visvol(imageio.imread('datasets/human/human_em_export_8nm/human_em_export_s0220.png'), \
	    imageio.imread('outputs/cluster_mask_3_sicidi_220.png'), filename='sicidi_3_em_220', ff='png', save=True, dpi=1200)
		if end:
			return

	def run(self):
		'''
		Running the main clustering algoritm on the features (feature list) extracted.
		'''
		labels, feat = self.get_features()
		clst_m = self.prep_cluster_matrix(labels, feat)
		res_labels = self.model.fit_predict(clst_m)
		_, gtfns = self.fe.get_fns()
		_ = recompute_from_res(labels, res_labels, volfns=gtfns, dprc=self.cfg.MODE.DPRC, fp=self.cfg.CLUSTER.OUTPUTPATH)

		print('\nfinished clustering.')

		# For visualization purposes.
		#labeled = imageio.imread(os.path.join(self.cfg.CLUSTER.OUTPUTPATH, 'cluster_mask_0.png'))
		#visvol(self.emvol[0], labeled)
