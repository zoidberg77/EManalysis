import h5py
import os, sys
import glob
import numpy as np
import imageio


def readh5(filename, dataset=''):
    # read the h5 file.
    hfile = h5py.File(filename, 'r')
    if dataset=='':
        dataset = list(hfile)[0]
    return np.array(hfile[dataset])

def readvol(filename, dataset=''):
    # read the volume in
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


def savelabvol(filename, vol, labels=None, dataset='main', format='h5'):
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
