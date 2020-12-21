import h5py
import os, sys
import glob
import numpy as np
import imageio


def readh5(filename, dataset=''):
	'''
	Read in the data volume in h5.
	:param filename: (string)
	:param dataset: (str) name of the volume.
	'''
	hfile = h5py.File(filename, 'r')
	if dataset=='':
		dataset = list(hfile)[0]
	return np.array(hfile[dataset])

def readvol(filename, dataset=''):
	'''
	Read in the data in different formats.
	:param filename: (string)
	:param dataset: (str) name of the volume.
	'''
	image = filename[filename.rfind('.')+1:]
	if image == 'h5':
		data = readh5(filename, dataset)
	elif 'tif' in image:
		data = imageio.volread(filename).squeeze()
	elif 'png' in image:
		data = readimgs(filename)
	else:
		raise ValueError('file format not found for %s'%(filename))
	return data

def readimgs(filename):
	'''
	Read images from folder.
	:param filename: (string)
	'''
	filelist = sorted(glob.glob(filename))
	num_imgs = len(filelist)

	img = imageio.imread(filelist[0])
	data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
	data[0] = img

	if num_imgs > 1:
		for i in range(1, num_imgs):
			data[i] = imageio.imread(filelist[i])

	return data


def folder2Vol(filepath, cv=None, file_format='png', maxF=-1, ratio=[1,1,1], fns=None, dt=np.uint16):
	'''
	Convert single image files (2D) to 3D h5 volume.
	:param filepath: (string) filepath
	:param cv: (list) defines the crop volume. e.g. [0, 1024, 0, 1024, 0, 100]
	:param file_format: (string) defines the input dataformat.
	'''
	if fns is None:
		fns = sorted(glob.glob(filepath + '*.' + file_format))

	numF = len(fns)
	if maxF > 0:
		numF = min(numF, maxF)

	numF = numF // ratio[0]

	sz = np.array(imageio.imread(fns[0]).shape)[:2] // ratio[1:]
	vol = np.zeros((numF, sz[0], sz[1]), dtype=dt)

	for zi in range(numF):
		if os.path.exists(fns[zi * ratio[0]]):
			tmp = imageio.imread(fns[zi * ratio[0]])
			if tmp.ndim == 3:
				tmp = tmp[:, :, 0]
				print(tmp)
			vol[zi] = tmp[::ratio[1], ::ratio[2]]

	if cv is not None:
		vol = vol[cv[4]:cv[5], cv[0]:cv[1], cv[2]:cv[3]]

	return vol

def savelabvol(vol, filename, labels=None, dataset='main', format='h5'):
	'''
	Save the data volume.
	:param filename: (string)
	:param vol: volume that will be saved.
	:param labels: groundtruth.
	:param dataset: (str) name of the volume.
	:param format: (str) dataformat.
	'''
	h5 = h5py.File(filename, 'w')
	#print(isinstance(dataset, (list, )))

	if isinstance(dataset, (list,)):
		for i,dd in enumerate(dataset):
			ds = h5.create_dataset(dd, vol[i].shape, compression="gzip", dtype=vol[i].dtype)
			ds[:] = vol[i]
			if labels is not None:
				ld = h5.create_dataset('labels', vol[i].shape, compression="gzip")
				ld[:] = labels
	else:
		ds = h5.create_dataset(dataset, vol.shape, compression="gzip", dtype=vol.dtype)
		ds[:] = vol
		if labels is not None:
			ld = h5.create_dataset('labels', vol.shape, compression="gzip")
			ld[:] = labels

	h5.close()
