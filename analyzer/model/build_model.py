import numpy as np
from sklearn.cluster import KMeans
from analyzer.model.utils.measuring import compute_regions, recompute_from_res, compute_intentsity
from analyzer.data.data_vis import visvol
from analyzer.utils.eval import clusteravg

class Clustermodel():
	'''
	Setups up the model for running a clustering algoritm on the loaded data.
	:param alg: choose how you want to cluster and label the segments.
				- 'bysize': cluster segements by their size.
				- 'bytext': cluster segments by texture. EM needed.

	:param mode: (string) Analyze either by 2d or 3d slizes.
	'''
	def __init__(self, emvol, gtvol, alg='bysize', mode='3d'):
		self.emvol = emvol
		self.gtvol = gtvol
		self.alg = alg
		self.mode = mode

		print(' --- model is set. --- ')

	def set_model(self, mn='kmeans'):
		'''
		This function enables the usage of different algoritms when setting the model overall.
		:param mn: (string) that is the name of the algoritm to go with.
		'''
		pass


	def run(self):
		if self.alg == 'bysize':
			labels, areas = compute_regions(self.gtvol, mode=self.mode)

			kmeans = KMeans(n_clusters=5)
			res_labels = kmeans.fit_predict(areas.reshape(-1,1))

			labeled = recompute_from_res(labels, res_labels, mode=self.mode)

			clmeans = clusteravg(areas, res_labels)
			print('means: ', clmeans)

			for k in range(labeled.shape[0]):
				visvol(self.emvol[k], labeled[k])

		elif self.alg == 'bytext':
			labels, intns = compute_intentsity(self.emvol, self.gtvol, mode='3d')

			kmeans = KMeans(n_clusters=5)
			res_labels = kmeans.fit_predict(intns.reshape(-1,1))

			labeled = recompute_from_res(labels, res_labels, mode=self.mode)

			clmeans = clusteravg(intns, res_labels)
			print('means: ', clmeans)

			for k in range(labeled.shape[0]):
				visvol(self.emvol[k], labeled[k])

		else:
			raise Exception('Please state according to which property should be clustered.')
