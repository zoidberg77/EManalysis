import numpy as np


class Dataloader():
	'''
	Dataloader class for handling the em dataset.

	:param
	'''
	def __init__(self,
                 volpath,
                 gtpath,
				 volume=None,
				 label=None,
                 sample_volume_size=(8, 64, 64),
                 sample_label_size=(8, 64, 64),
                 ):

		if volume is None:
            pass
        else:
            self.volume = volume

        if label is None:
            pass
        else:
            self.label = label

        self.sample_volume_size = sample_volume_size
        self.sample_label_size = sample_label_size

	def load(self):
		pass

	def folder2vol(self):
		'''
		Convert single image files (2D) to 3D h5 volume.
		:param filepath: (string) filepath
		:param cv: (list) defines the crop volume. e.g. [0, 1024, 0, 1024, 0, 100]
		:param file_format: (string) defines the input dataformat.
		'''
        fns = sorted(glob.glob(filepath + '*.' + file_format))

		numF = len(fns)
		if maxF > 0:
			numF = min(numF, maxF)
		numF = numF // ratio[0]

		sz = np.array(imread(fns[0]).shape)[:2] // ratio[1:]
		vol = np.zeros((numF, sz[0], sz[1]), dtype=dt)

		for zi in range(numF):
			if os.path.exists(fns[zi * ratio[0]]):
				tmp = imread(fns[zi * ratio[0]])
				if tmp.ndim == 3:
					tmp = tmp[:, :, 0]
				vol[zi] = tmp[::ratio[1], ::ratio[2]]

		vol = vol[cv[4]:cv[5], cv[0]:cv[1], cv[2]:cv[3]]

		return vol
