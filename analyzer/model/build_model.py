import os, sys
import numpy as np
from skimage.measure import label
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN
from analyzer.model.utils.extracting import compute_region_size, compute_intentsity, compute_dist_graph
from analyzer.model.utils.superpixel import superpixel_segment, superpixel_image, texture_analysis
from analyzer.model.utils.helper import convert_to_sparse, recompute_from_res, convert_dict_mtx
from analyzer.data.data_vis import visvol, vissegments
from analyzer.utils.eval import clusteravg

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
	def __init__(self, emvol, gtvol, dl=None, alg='kmeans', clstby='bysize', n_cluster=5, mode='3d'):
		self.emvol = emvol
		self.gtvol = gtvol
		self.dl = dl
		self.alg = alg
		self.clstby = clstby
		self.n_cluster = n_cluster
		self.mode = mode

		self.model = self.set_model(mn=self.alg)

		print(' --- model is set. algorithm: {}, clustering: {} --- '.format(self.alg, self.clstby))

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

	def load_features(self, fpath=os.path.join(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__)), 'features/')):
		'''
		This function will load different features vectors that were extracted and saved to be used for clustering.
		'''
		if os.path.exists(fpath) is False:
			raise ValueError('Please enter a valid path to the folder where the feature vectors are stored.')

		print(fpath)

	def stack_features(self):
		'''
		This function takes different features and stacks them togehter for further clustering.
		'''
		raise NotImplementedError


	def run(self):
		if self.clstby == 'bysize':
			# RUN the clustering by size parameters.
			rst_dict = compute_region_size(self.gtvol, mode=self.mode)
			labels, areas = convert_dict_mtx(rst_dict)

			res_labels = self.model.fit_predict(areas.reshape(-1,1))
			labeled = recompute_from_res(self.gtvol, labels, res_labels, mode=self.mode)

			clmeans = clusteravg(areas, res_labels)
			print('means: ', clmeans)

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

			#test = self.dl.prep_data_info()
			#print(test)

			rst_dict = compute_dist_graph(self.gtvol, mode=self.mode)
			labels, dist_m = convert_dict_mtx(rst_dict)

			res_labels = self.model.fit_predict(dist_m)
			labeled = recompute_from_res(self.gtvol, labels, res_labels, mode=self.mode)

			for k in range(self.emvol.shape[0]):
				visvol(self.emvol[k], labeled[k])

		else:
			raise Exception('Please state according to which property should be clustered.')
