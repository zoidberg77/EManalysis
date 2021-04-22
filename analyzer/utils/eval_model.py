import os, sys
import numpy as np
import json
import glob
import multiprocessing
import functools
from numpyencoder import NumpyEncoder
import imageio

from sklearn.metrics import normalized_mutual_info_score, pair_confusion_matrix
from analyzer.model.utils.extracting import calc_props

class Evaluationmodel():
	'''
	Setups up the model for evaluation purposes after the clustering is finished
	and a groundtruth is there. You might also just decide to not use it in order to
	keep it unsupervised.
	:param cfg: configuration manager.
	:param dl: Dataloader
	:param rsl_vector: This is the resulting vector extracted from the clustering model.
					   (n,) (np.array) with n beeing the number of samples.
	'''
	def __init__(self, cfg, dl):
		self.cfg = cfg
		self.dl = dl
		#self.rsl_vector = rsl_vector

	def eval(self, rsl_vector):
		'''
		Evaluation of the clustering by comparing the gt to the results.
		'''
		score = normalized_mutual_info_score(self.get_gt_vector(), rsl_vector)
		print(score)

	def get_gt_vector(self, fn='gt_vector.json'):
		return self.create_gt_vector()

	def create_gt_vector(self, fn='gt_vector.json', save=True):
		'''
		This function should create a resulting label vector that is the ground truth.
		:returns (n,) vector. n is the number of samples/segments.
		'''
		if os.path.exists(os.path.join(self.cfg.DATASET.ROOTF, fn)) \
				and os.stat(os.path.join(self.cfg.DATASET.ROOTF, fn)).st_size != 0:
			with open(os.path.join(self.cfg.DATASET.ROOTF, fn), 'r') as f:
				gt_vector = json.loads(f.read())
		else:
			print('gt vector not found. Will be computed.')
			if os.path.exists(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.ROOTF, 'eval_data_info.json')) \
					and os.stat(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.ROOTF, 'eval_data_info.json')).st_size != 0:
				with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.ROOTF, 'eval_data_info.json'), 'r') as f:
					data_info = json.loads(f.read())
			else:
				print('data info not found. Will be computed.')
				data_info = self.prep_data_info(save=True)

			fns = sorted(glob.glob(self.dl.gtpath + '*.' + self.cfg.DATASET.FILE_FORMAT))
			gt_ids = list(map(int, data_info.keys()))
			gt_vector = np.zeros(len(gt_ids), dtype=np.uint16)

			for key, value in data_info.items():
				slices = value[0]
				centerpoints = value[1]
				randompts = value[2]
				for i, s in enumerate(slices):
					gt = imageio.imread(fns[s])
					if gt[centerpoints[i][0], centerpoints[i][1]] == 0 and gt[randompts[i][0], randompts[i][1]] == 0:
						continue
					else:
						if gt[centerpoints[i][0], centerpoints[i][1]] != 0:
							gt_vector[gt_ids.index(int(key))] = gt[centerpoints[i][0], centerpoints[i][1]]
						else:
							gt_vector[gt_ids.index(int(key))] = gt[randompts[i][0], randompts[i][1]]
						break

				if gt_ids.index(int(key)) % 1000 == 0:
					print('altered [{}/{}] labels for ground truth vector.'.format(gt_ids.index(int(key)), len(gt_ids)))
			if save:
				with open(os.path.join(self.cfg.DATASET.ROOTF, 'gt_vector.json'), 'w') as f:
					json.dump(gt_vector, f, cls=NumpyEncoder)
					f.close()

		values, counts = np.unique(gt_vector, return_counts=True)
		if (values == 0).any():
			print('gt vector contains 0 as label.')
			print('values: ', values)
			print('counts: ', counts)

		return gt_vector

	def prep_data_info(self, save=False):
		'''
		Extracting the label and its centerpoints.
		'''
		fns = sorted(glob.glob(self.dl.labelpath + '*.' + self.cfg.DATASET.FILE_FORMAT))

		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			tmp = pool.starmap(functools.partial(calc_props, prop_list=['slices', 'centroid', 'random_pt']), enumerate(fns))

		result_dict = {}
		for dicts in tmp:
			for key, value in dicts.items():
				if key in result_dict:
					result_dict[key][0].append(value[0])
					result_dict[key][1].append(value[1])
					result_dict[key][2].append(value[2])
				else:
					result_dict.setdefault(key, [])
					result_dict[key].append([value[0]])
					result_dict[key].append([value[1]])
					result_dict[key].append([value[2]])

		if save:
			with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.ROOTF, 'eval_data_info.json'), 'w') as f:
				json.dump(result_dict, f, cls=NumpyEncoder)
				f.close()
		return result_array
