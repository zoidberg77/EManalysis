import numpy as np
from skimage.segmentation import slic
from skimage.exposure import equalize_hist

from analyzer.data.data_vis import visvol, vissegments

def superpixel_segment(vol, gt, mode='2d'):
    '''
    This function computes superpixels within every segment.
    :param vol: volume (np.array) that contains the bare em data. (2d || 3d)
	:param gt: volume (np.array) that contains the groundtruth. (2d || 3d)
    '''
    mask = np.zeros(shape=vol.shape, dtype=np.uint16)
    mask[gt > 0] = 1
    vol[mask == 0] = 0
    eqvol = equalize_hist(vol)

    if mode == '2d':
        for idx in range(vol.shape[0]):
            mask2d = mask[idx]
            segments = slic(vol[idx], n_segments=10, mask=mask2d, start_label=1, compactness=.01)
            #segments = slic(vol[idx], n_segments=200, start_label=1, compactness=.1)
    else:
        raise NotImplementedError('no 3d mode in this function yet.')

    vissegments(vol[0], segments[0], mask=mask[0])

    return segments
