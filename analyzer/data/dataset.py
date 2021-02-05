import h5py
import numpy as np
import os, sys
import glob
import imageio
import multiprocessing
from skimage.measure import label, regionprops
from analyzer.data.data_vis import visvol
from analyzer.data.data_raw import readvol, folder2Vol
import torch
import torchio as tio


class Dataloader():
	'''
	Dataloader class for handling the em dataset and the related labels.

	:param volpath: (string) path to the directory that contains the em volume(s).
	:param gtpath: (string) path to the directory that contains the groundtruth volume(s).
	:param volume:
	:param label:
	:param chunk_size: (tuple) defines the chunks in which the data is loaded. Can help to overcome Memory errors.
	:param mode: (string) Sets the mode in which the volume should be clustered (3d || 2d).
	:param ff: (string) defines the file format that you want to work with. (default: png)
	'''
	def __init__(self,
				 volpath,
				 gtpath,
				 volume=None,
				 label=None,
				 chunk_size=(100,4096,4096),
				 mode='3d', ff='png'
				 ):

		if volume is not None:
			pass
		else:
			self.volpath = volpath
			self.volume = volume

		if label is not None:
			pass
		else:
			self.gtpath = gtpath
			self.label = label

		self.chunk_size = chunk_size
		self.mode = mode
		self.ff = ff

	def __len__(self):
		'''
		Required by torch to return the length of the dataset.
		:returns: integer
		'''
		return self.volume.shape[0]

	def __getitem__(self, idx):
		'''
		Required by torch to return one item of the dataset.
		:param idx: index of the object
		:returns: object from the volume
		'''
		return self.volume[idx]

	def load_chunk(self, vol='both'):
		'''
		Load chunk of em and groundtruth data for further processing.
		:param vol: (string) choose between -> 'both', 'em', 'gt' in order to specify
					 with volume you want to load.
		'''
		emfns = sorted(glob.glob(self.volpath + '*.' + self.ff))
		gtfns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
		emdata = 0
		gt = 0

		if self.mode == '2d':
			if (vol == 'em') or (vol == 'both'):
				emdata = readvol(emfns[0])
				emdata = np.squeeze(emdata)
				print('em data loaded: ', emdata.shape)
			if (vol == 'gt') or (vol == 'both'):
				gt = readvol(gtfns[0])
				gt = np.squeeze(gt)
				print('gt data loaded: ', gt.shape)

		if self.mode == '3d':
			if (vol == 'em') or (vol == 'both'):
				if self.volume is None:
					emdata = folder2Vol(self.volpath, self.chunk_size, file_format=self.ff)
					print('em data loaded: ', emdata.shape)
			if (vol == 'gt') or (vol == 'both'):
				if self.label is None:
					gt = folder2Vol(self.gtpath, self.chunk_size, file_format=self.ff)
					print('gt data loaded: ', gt.shape)

		return (emdata, gt)

	def list_segments(self, vol, labels, min_size=2000, os=0, mode='3d'):
		'''
		This function creats a list of arrays that contain the unique segments.
		:param vol: (np.array) volume that contains the pure em data. (2d || 3d)
		:param label: (np.array) volume that contains the groundtruth. (2d || 3d)
		:param min_size: (int) this sets the minimum size of mitochondria region in order to be safed to the list. Used only in 2d.
		:param os: (int) defines the offset that should be used for cutting the bounding box. Be careful with offset as it can lead to additional regions in the chunks.
		:param mode: (string) 2d || 3d --> 2d gives you 2d arrays of each slice (same mitochondria are treated differently as they loose their touch after slicing)
									   --> 3d gives you the whole mitochondria in a 3d volume.

		:returns: (dict) of (np.array) objects that contain the segments with labels as keys.
		'''
		bbox_dict = {}

		mask = np.zeros(shape=vol.shape, dtype=np.uint16)
		mask[labels > 0] = 1
		vol[mask == 0] = 0

		if mode == '2d':
			bbox_list = []
			for idx in range(vol.shape[0]):
				image = vol[idx, :, :]
				gt_img = labels[idx, :, :]
				label2d, num_label = label(gt_img, return_num=True)
				regions = regionprops(label2d, cache=False)

				for props in regions:
					boundbox = props.bbox
					if props.bbox_area > min_size:
						if ((boundbox[0] - os) < 0) or ((boundbox[2] + os) > image.shape[0]) or ((boundbox[1] - os) < 0) or ((boundbox[3] + os) > image.shape[1]):
							tmparr = image[boundbox[0]:boundbox[2], boundbox[1]:boundbox[3]]
						else:
							tmparr = image[(boundbox[0] - os):(boundbox[2] + os), (boundbox[1] - os):(boundbox[3] + os)]

						bbox_list.append(tmparr)

			bbox_dict = { i : bbox_list[i] for i in range(len(bbox_list)) }

		elif mode == '3d':
			chunk_dict = {}

			label3d, num_label = label(labels, return_num=True)
			regions = regionprops(label3d, cache=False)

			for props in regions:
				boundbox = props.bbox

				if ((boundbox[1] - os) < 0) or ((boundbox[4] + os) > vol.shape[1]) or ((boundbox[2] - os) < 0) or ((boundbox[5] + os) > vol.shape[2]):
					tmparr = vol[boundbox[0]:boundbox[3], boundbox[1]:boundbox[4], boundbox[2]:boundbox[5]]
				else:
					tmparr = vol[boundbox[0]:boundbox[3], (boundbox[1] - os):(boundbox[4] + os), (boundbox[2] - os):(boundbox[5] + os)]

				bbox_dict[props.label] = tmparr

		else:
			raise ValueError('No valid dimensionality mode in function list_segments.')

		return (bbox_dict)

	def prep_isotropic_seg(self, vol, gt):
		'''
		This function prepars the input segments for the autoencoder. Isotropic and equal size.
		'''
		label3d, num_label = label(gt, return_num=True)
		regions = regionprops(label3d, cache=False)

		c_x_max = 0
		c_y_max = 0
		c_z_max = 0
		for props in regions:
			boundbox = props.bbox

			if boundbox[4] - boundbox[1] >= c_x_max:
				c_x_max = boundbox[4] - boundbox[1]
			if boundbox[3] - boundbox[0] >= c_y_max:
				c_y_max = boundbox[3] - boundbox[0]
			if boundbox[5] - boundbox[2] >= c_z_max:
				c_z_max = boundbox[5] - boundbox[2]

		print(c_x_max, ' ', c_y_max, ' ', c_z_max)

	def prep_data_info(self, volopt='gt', kernel_n=multiprocessing.cpu_count()):
		'''
		This function aims as an inbetween function iterating over the whole dataset in efficient
		and memory proof fashion in order to preserve information that is needed for further steps.
		:param volopt: (string) this sets the volume you want to use for the operation. default: gt
		:param kernel_n: (int) number of CPU kernels you want to use for multiprocessing.

		:returns added: (dict) that contains the labels with respective information as (list): [pixelsize, [slice_index(s)]]
		'''
		if volopt == 'gt':
			fns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
		elif volopt == 'em':
			fns = sorted(glob.glob(self.volpath + '*.' + self.ff))
		else:
			raise ValueError('Please enter the volume on which \'prep_data_info\' should run on.')

		with multiprocessing.Pool(processes=kernel_n) as pool:
			result = pool.starmap(self.calc_props, enumerate(fns))

		added = {}
		for dicts in result:
			for key, value in dicts.items():
				#print(value)
				if key in added:
					added[key][0] += value[0]
					added[key][1].append(value[1])
				else:
					added.setdefault(key, [])
					added[key].append(value[0])
					added[key].append([value[1]])

		result_array = []
		for result in added.keys():
			result_array.append({
				'id': result,
				'size': added[result][0],
				'slices': added[result][1]
			})

		#print("Max values: ", max([x[0] for x in added.values()]))
		return (result_array)

	def calc_props(self, idx, fns):
		'''
		Helper function for 'prep_data_info'
		:param idx: (int) this is the slice index that correspondes to the image slice. E.g. idx 100 belongs to image 100.
		:param fns: (string) list of filenames.
		:returns result: (dict) with each segment. key: idx of segment -- value: [number of pixels in segment, idx of slice].
		'''
		result = {}
		idx_list = []
		if os.path.exists(fns):
			tmp = imageio.imread(fns)
			labels, num_labels = np.unique(tmp, return_counts=True)

			for l in range(labels.shape[0]):
				if labels[l] == 0:
					continue
				result.setdefault(labels[l], [])
				result[labels[l]].append(num_labels[l])
				result[labels[l]].append(idx)

		return result

	def get_mito_volume(self, region, target=(1, 64, 64, 64)):
		'''
		Preprocessing function to extract and scale the mitochondria as volume
		:param region: (dict) one region object provided by Dataloader.prep_data_info
		:param target: (int,int,int,int) these are the target dimensions e.g. (1,64,64,64)
		:returns result: (numpy.array) a numpy array with the target dimensions and the mitochondria in it
		'''
		all_fn = sorted(glob.glob(self.gtpath + '*.' + self.ff))
		target = tio.ScalarImage(tensor=torch.rand(target))
		fns = [all_fn[id] for id in region['slices']]
		first_image_slice = imageio.imread(fns[0])
		mask = np.zeros(shape=first_image_slice.shape, dtype=np.uint16)
		mask[first_image_slice == region['id']] = 1
		volume = mask
		for fn in fns[1:]:
			image_slice = imageio.imread(fn)
			mask = np.zeros(shape=image_slice.shape, dtype=np.uint16)
			mask[first_image_slice == region['id']] = 1
			volume = np.dstack((volume, mask))
		volume = np.moveaxis(volume,-1,0)

		mito_regions = regionprops(volume, cache=False)
		if len(mito_regions) != 1:
			print("something went wrong during volume building. region count: {}".format(len(mito_regions)))

		mito_region = mito_regions[0]
		mito_volume = volume[mito_region.bbox[0]:mito_region.bbox[3]+1,
					 mito_region.bbox[1]:mito_region.bbox[4]+1,
					 mito_region.bbox[2]:mito_region.bbox[5]+1].astype(np.float32)

		mito_volume = np.expand_dims(mito_volume, 0)
		transform = tio.Resample(target=target, image_interpolation='bspline')
		transformed_mito = transform(mito_volume)
		return transformed_mito

	def save_mito_volume_to_h5(self, volume, filename="mito.h5", dset_name="mito_volumes"):
		f = None
		dset = None
		dims = volume.shape

		if os.path.exists(filename):
			f = h5py.File('mito.h5', 'a')
		else:
			f = h5py.File('mito.h5', 'w')

		if dset_name in f.keys():
			dset = f[dset_name]
		else:
			dset = f.create_dataset(dset_name, (0, dims[0], dims[1], dims[2], dims[3]), maxshape=(None, dims[0], dims[1], dims[2], dims[3]))

		dset.resize((dset.shape[0] + 1, dset.shape[1], dset.shape[2], dset.shape[3], dset.shape[4]))
		dset[-1] = volume