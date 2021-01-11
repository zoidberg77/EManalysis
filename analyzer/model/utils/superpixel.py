import numpy as np
from skimage.segmentation import slic

def slic_segment(vol, gt):
    '''
    This function computes superpixels within every segment.
    :param vol: volume (np.array) that contains the bare em data. (2d || 3d)
	:param gt: volume (np.array) that contains the groundtruth. (2d || 3d)
    '''
    pass
