import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN
from analyzer.model.utils.measuring import compute_regions, recompute_from_res, compute_intentsity
from analyzer.model.utils.superpixel import superpixel_segment
from analyzer.data.data_vis import visvol, vissegments
from analyzer.utils.eval import clusteravg

class Clustermodel():
	'''
	Setups up the model for running a clustering algoritm on the loaded data.
	:param emvol & gtvol: (np.array) Both are the data volumes.
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
	def __init__(self, emvol, gtvol, alg='kmeans', clstby='bysize', n_cluster=5, mode='3d'):
		self.emvol = emvol
		self.gtvol = gtvol
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


	def run(self):
		if self.clstby == 'bysize':
			labels, areas = compute_regions(self.gtvol, mode=self.mode)

			res_labels = self.model.fit_predict(areas.reshape(-1,1))

			labeled = recompute_from_res(labels, res_labels, mode=self.mode)

			clmeans = clusteravg(areas, res_labels)
			print('means: ', clmeans)

			for k in range(labeled.shape[0]):
				visvol(self.emvol[k], labeled[k])

		elif self.clstby == 'bytext':
			labels, intns = compute_intentsity(self.emvol, self.gtvol, mode='3d')

			res_labels = self.model.fit_predict(intns.reshape(-1,1))

			#segments = superpixel_segment(self.emvol, self.gtvol)


			#labeled = recompute_from_res(labels, res_labels, mode=self.mode)

			#clmeans = clusteravg(intns, res_labels)
			#print('means: ', clmeans)

			#for k in range(self.emvol.shape[0]):
				#visvol(self.emvol[k], labeled[k])
				#vissegments(self.emvol[k], segments[k])

		else:
			raise Exception('Please state according to which property should be clustered.')
