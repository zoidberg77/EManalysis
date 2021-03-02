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

def folder2Vol(filepath, chunk_size=None, fns=None, file_format='png', dt=np.uint16):
	'''
	Convert single image files (2D) to 3D h5 volume.
	:param filepath: (string) filepath
	:param chunk_size: (tuple) that will define the size of different chunks.
	:param fns: (string) defines the filenames.
	:param file_format: (string) defines the input dataformat.
	:param dt: (dataformat) default: np.uint16

	:returns: numpy array that contains the raw data (em & labels).
			  shape: (500, 4096, 4096)
			  Chunks that part the images will look like this (4, 100, 2048, 2048).
	'''
	if fns is None:
		fns = sorted(glob.glob(filepath + '*.' + file_format))

	if len(fns) == 0:
		raise ValueError("Please enter valid filepath.")

	sz = np.array(imageio.imread(fns[0]).shape)[:2]

	if chunk_size is None:
		vol = np.zeros((len(fns), sz[0], sz[1]), dtype=dt)
		for zi in range(len(fns)):
			if os.path.exists(fns[zi]):
				tmp = imageio.imread(fns[zi])
				if tmp.ndim >= 3:
					tmp = np.squeeze(tmp)
				vol[zi] = tmp
	else:
		ratio = sz[0] / chunk_size[1]
		if sz[0] % ratio != 0:
			raise ValueError("Consider a different chunk size as the ratio does not fit the original image size. Keep it squared.")

		ratio = int(ratio)
		if ratio != 1:
			vol = np.zeros((ratio * 2, chunk_size[0], chunk_size[1], chunk_size[2]), dtype=dt)
		else:
			vol = np.zeros((chunk_size[0], chunk_size[1], chunk_size[2]), dtype=dt)

		for zi in range(chunk_size[0]):
			if os.path.exists(fns[zi]):
				tmp = imageio.imread(fns[zi])
				if tmp.ndim >= 3:
					tmp = np.squeeze(tmp)

				if ratio != 1:
					splitarr = []
					htmp = np.hsplit(tmp, ratio)
					for j in range(ratio):
						vtmp = np.vsplit(htmp[j], ratio)
						for elements in range(ratio):
							splitarr.append(vtmp[elements])

					for i in range(ratio * 2):
						vol[i, zi, :, :] = splitarr[i]
				else:
					vol[zi] = tmp

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

def save_m_to_image(img, filename, fp, idx=None, ff='png'):
	'''
	Save the data volume.
	:param filename: (string)
	:param img: 2d (np.array) that contains the information you want to save.
	'''
	if idx is not None:
		fn = filename + '_' + str(idx) + '.' + ff
	else:
		fn = filename + '.' + ff
	imageio.imwrite(os.path.join(fp, fn), img.astype(np.uint8))
	print('saving resulting image as {} to {}'.format(fn, fp))
