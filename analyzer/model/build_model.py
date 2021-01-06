import numpy as np
from sklearn.cluster import KMeans
from analyzer.model.utils.measuring import compute_regions, recompute_from_res
from analyzer.data.data_vis import visvol
from analyzer.utils.eval import clusteravg

class Clustermodel():
	'''
	Setups up the model for running a clustering algoritm on the loaded data.
	:param alg: choose how you want to cluster and label the segments.
				- 'bysize':
	'''
	def __init__(self, emvol, gtvol, alg='bysize'):
		self.emvol = emvol
		self.gtvol = gtvol
		self.alg = alg

		print('model is set.')

	def run(self):
		if self.alg == 'bysize':
			labels, areas = compute_regions(self.gtvol)
			kmeans = KMeans(n_clusters=5)
			res_labels = kmeans.fit_predict(areas.reshape(-1,1))

			labeled = recompute_from_res(labels, res_labels)

			clmeans = clusteravg(areas, res_labels)

			print('means: ', clmeans)

			for k in range(labeled.shape[0]):
				visvol(self.emvol[k], labeled[k])
