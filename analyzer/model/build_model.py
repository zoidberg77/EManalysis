import numpy as np
from sklearn.cluster import KMeans
from analyzer.model.utils.measuring import compute_regions
from analyzer.data.data_vis import visvol

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

	def run(self):
		if self.alg == 'bysize':
			labels, areas = compute_regions(self.gtvol)
			kmeans = KMeans()
			new_labels = kmeans.fit_predict(areas)

			#visvol()
