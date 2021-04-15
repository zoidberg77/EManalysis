import os, sys
import numpy as np
import json
import glob
import multiprocessing
import functools
from numpyencoder import NumpyEncoder
import imageio

from analyzer.model.utils.extracting import calc_props

class Evaluationmodel():
	'''
	Setups up the model for evaluation purposes after the clustering is finished
	and a groundtruth is there. You might also just decide to not use it in order to
	keep it unsupervised.
	:param cfg: configuration manager.
	:param dl: Dataloader
	'''
	def __init__(self, cfg, dl):
		self.cfg = cfg
		self.dl = dl

	def load_gt_vector(self, fn='gt_vector.json'):
		if os.path.exists(os.path.join(self.cfg.DATASET.ROOTF, fn)) \
				and os.stat(os.path.join(self.cfg.DATASET.ROOTF, fn)).st_size != 0:
			with open(os.path.join(self.cfg.DATASET.ROOTF, fn), 'r') as f:
				gt_vector = json.loads(f.read())
		return gt_vector

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
			if os.path.exists(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)) \
					and os.stat(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)).st_size != 0:
				with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'r') as f:
					data_info = json.loads(f.read())
			else:
				data_info = self.prep_data_info(save=True)

			fns = sorted(glob.glob(self.dl.gtpath + '*.' + self.cfg.DATASET.FILE_FORMAT))
			gt_ids = list(map(int, data_info.keys()))
			gt_vector = np.zeros(len(gt_ids), dtype=np.uint16)

			for key, value in data_info.items():
				slices = value[0]
				centerpoints = value[1]
				gt = imageio.imread(fns[slices[0]])

				gt_vector[gt_ids.index(int(key))] = gt[centerpoints[0][0], centerpoints[0][1]]

				if gt_ids.index(int(key)) % 1000 == 0:
					print('altered {} labels for ground truth vector of {} labels.'.format(gt_ids.index(int(key)), len(gt_ids)))
			if save:
				with open(os.path.join(self.cfg.DATASET.ROOTF, 'gt_vector.json'), 'w') as f:
					json.dump(gt_vector, f, cls=NumpyEncoder)
					f.close()


		#### Testing part #####
		if os.path.exists(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)) \
				and os.stat(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)).st_size != 0:
			with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'r') as f:
				data_info = json.loads(f.read())
		else:
			data_info = self.prep_data_info(save=True)

		fns = sorted(glob.glob(self.dl.gtpath + '*.' + self.cfg.DATASET.FILE_FORMAT))
		gt_ids = list(map(int, data_info.keys()))
		for key, value in data_info.items():
			if gt_vector[gt_ids.index(int(key))] == 0:
				slices = value[0]
				centerpoints = value[1]
				gt = imageio.imread(fns[slices[1]])

				gt_vector[gt_ids.index(int(key))] = gt[centerpoints[1][0], centerpoints[1][1]]
			else:
				continue

		if save:
			with open(os.path.join(self.cfg.DATASET.ROOTF, 'gt_vector.json'), 'w') as f:
				json.dump(gt_vector, f, cls=NumpyEncoder)
				f.close()

		values, counts = np.unique(gt_vector, return_counts=True)
		print(values)
		print(counts)

	def prep_data_info(self, save=False):
		'''
		Extracting the label and its centerpoints.
		'''
		fns = sorted(glob.glob(self.dl.labelpath + '*.' + self.cfg.DATASET.FILE_FORMAT))

		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			tmp = pool.starmap(functools.partial(calc_props, prop_list=['slices', 'centroid']), enumerate(fns))

		result_dict = {}
		for dicts in tmp:
			for key, value in dicts.items():
				if key in result_dict:
					result_dict[key][0].append(value[0])
					result_dict[key][1].append(value[1])
				else:
					result_dict.setdefault(key, [])
					result_dict[key].append([value[0]])
					result_dict[key].append([value[1]])

		result_array = []
		for result in result_dict.keys():
			result_array.append({
				'id': result,
				'slices': result_dict[result][0],
				'ctps': result_dict[result][1]
			})

		if save:
			with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'w') as f:
				json.dump(result_dict, f, cls=NumpyEncoder)
				f.close()
		return result_array
