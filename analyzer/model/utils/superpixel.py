import numpy as np
from skimage.segmentation import slic
from skimage.exposure import equalize_hist

from analyzer.data.data_vis import visvol, vissegments

def superpixel_image(vol, gt, mode='2d'):
    '''
    This function computes superpixels within every segment of the whole image and returns the segments.
    :param vol: volume (np.array) that contains the bare em data. (2d || 3d)
	:param gt: volume (np.array) that contains the groundtruth. (2d || 3d)

    :returns segments: ()
    '''
    #TODO --> This function is not complete and used.

    mask = np.zeros(shape=vol.shape, dtype=np.uint16)
    mask[gt > 0] = 1
    vol[mask == 0] = 0
    eqvol = equalize_hist(vol)

    if mode == '2d':
        for idx in range(vol.shape[0]):
            mask2d = mask[idx]
            segments = slic(vol[idx], n_segments=10, mask=mask2d, start_label=1, compactness=10)
            #segments = slic(vol[idx], n_segments=200, start_label=1, compactness=.1)
    else:
        raise NotImplementedError('no 3d mode in this function yet.')

    vissegments(vol[0], segments[0], mask=mask[0])

    return segments


def superpixel_segment(segments, n_seg=10):
    '''
    This function computes superpixels within every segment.
    :param segments: (list) object containing (np.array)s that are the mitochondria segments.
    :param n_seg: (int) defines the approximate number of segments the slic should find.
    '''

    if segments[0].ndim == 2:
        raise NotImplementedError('no 2d mode in this function yet.')
    else:
        for idx in range(len(segments)):
            seg = segments[idx]
            # create a mask.
            mask = np.zeros(seg.shape, dtype=np.uint16)
            mask[seg > 0] = 1

            slice = seg[0]
            slic_res = slic(slice, n_segments=n_seg, compactness=0.01, mask=mask[0], start_label=1)

            #vissegments(slice, slic_res, mask=mask[0])

    #return segments
