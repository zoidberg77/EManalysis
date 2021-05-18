import os, sys
import json
import copy
import multiprocessing
import functools
import numpy as np
from numpyencoder import NumpyEncoder
from analyzer.model.utils.extracting import calc_props

### --- running example ---
# import glob
# from analyzer.utils.misc import find_cluster_center
# from analyzer.utils.eval_model import Evaluationmodel
#
# gt_fns = sorted(glob.glob('datasets/mouseA/mito_export_maingroups/' + '*.' + 'png'))
# unique_fns = sorted(glob.glob('datasets/mouseA/mito_export_unique_id/' + '*.' + 'png'))
#
# eval = Evaluationmodel(cfg, dl=dl)
# find_cluster_center(eval, cfg, unique_fns, gt_fns, save=True)

def find_cluster_center(eval_model, cfg, unique_fns, gt_fns, save=False):
	'''
	This function computes an average centerpoint of a cluster and returns it.
	Additionally, the nearest segment & label is determined and returned as well.
	:param eval_model: class object from analyzer.utils.eval_model
	:params cfg: configuration file.
	:param unique_fns: sorted (list) that contains filenames of all the uniquely labeled segments.
	:param gt_fns: sorted (list) that contains filenames of all the gt and clustered segments.
	'''
	if os.path.exists(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.DATASET.ROOTF, 'mito_centroids.json')) \
			and os.stat(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.DATASET.ROOTF, 'mito_centroids.json')).st_size != 0:
		with open(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.DATASET.ROOTF, 'mito_centroids.json'), 'r') as f:
			centroids = json.loads(f.read())
	else:
		print('centroids not found. Will be computed. This takes a while. Be prepared.')
		centroids = compute_centerpoints(cfg, gt_fns, save=True)

	gt_labels = [x['id'] for x in centroids]
	gt_cpts = [x['c'] for x in centroids]

	unique_cs = compute_centerpoints(cfg, unique_fns, save=False)
	gt_vector = eval_model.get_gt_vector()

	result_dict = dict.fromkeys(gt_labels, dict.fromkeys(['dist', 'label', 'c'], None))
	for gt in gt_labels:
		result_dict[gt]['dist'] = 10000.0

	for idx, gt in enumerate(gt_vector):
		seg = unique_cs[idx]

		dist = np.linalg.norm(np.array(gt_cpts[gt_labels.index(gt)])-np.array(seg['c']))

		if dist < result_dict[gt]['dist']:
			tmp_dict = copy.deepcopy(result_dict)
			tmp_dict[gt]['dist'] = dist
			tmp_dict[gt]['label'] = seg['id']
			tmp_dict[gt]['c'] = seg['c']

		result_dict[gt] = tmp_dict[gt]

	if save:
		with open(os.path.join('central_mitochondria.json'), 'w') as f:
			json.dump(result_dict, f, cls=NumpyEncoder)
			f.close()
	print('resulting central mitochondrias: ', result_dict)
	return result_dict

def compute_centerpoints(cfg, fns, save=True):
	'''Compute centroids of every segment.'''
	result_dict = {}
	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		tmp = pool.starmap(functools.partial(calc_props, prop_list=['size', 'slices', 'centroid']), enumerate(fns))

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

	labels = list(result_dict.keys())
	centerpts = list()
	for key, value in result_dict.items():
		pt = list(map(int, [sum(x) / len(x) for x in zip(*value[2])]))
		tmp_z = 0.0
		for i in range(len(value[0])):
			tmp_z += (value[0][i] / sum(value[0])) * value[1][i]
		pt.append(int(tmp_z))
		centerpts.append(pt)

	centerpts = np.array(centerpts, dtype=np.int16)
	for idx in range(len(labels)):
		result_dict[labels[idx]].append([centerpts[idx]])

	result_array = []
	for result in result_dict.keys():
		result_array.append({
			'id': result,
			'c': result_dict[result][3],
		})
	if save:
		with open(os.path.join(cfg.SYSTEM.ROOT_DIR, cfg.DATASET.ROOTF, 'mito_centroids.json'), 'w') as f:
			json.dump(result_array, f, cls=NumpyEncoder)
			f.close()

	print('centerpoints computation finished. {} extracted.'.format(len(result_array)))
	return (result_array)


def update_nested(in_dict, key, value):
   for k, v in in_dict.items():
	   if key == k:
		   in_dict[k] = value
	   elif isinstance(v, dict):
		   update_nested(v, key, value)
	   elif isinstance(v, list):
		   for o in v:
			   if isinstance(o, dict):
				   update_nested(o, key, value)
