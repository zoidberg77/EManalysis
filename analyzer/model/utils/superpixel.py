import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import slic
from skimage.exposure import equalize_hist
from skimage.feature import greycomatrix, greycoprops

from analyzer.data.data_vis import visvol, vissegments, visbbox

def texture_analysis(segments, mode='3d', method='slic'):
    '''
    This function analysis the texture in the segments.
    :param segments: (list) object containing (np.array)s that are the mitochondria segments.
    :param mode: (string)
    :param method: (string) Differentiate between the amount of information you want to extract.
                            - 'fast':
                            - 'sliding_window':
                            - 'slic':
    :returns texts: (dict) of (np.array)s that contain the correlation values of each segments.
                    Keys represent the labels. Values represent the corr values vector.
                    Correlation values are extracted from a GLCM.
                    Check https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycomatrix
    '''
    #TODO: For the 3d mode --> Make it truly 3d as you are doing slic in 2d and just expand it to 3d.
    print('number of segments (mitochondria): ', len(segments))
    if mode == '2d':
        raise NotImplementedError('no 2d mode in this function yet.')
    elif mode == '3d':
        texts = {}
        for idx in range(len(segments)):
            vol = segments[idx + 1] # Plus 1 as dict starts with 1 as key label 1 instead of 0.
            corr_value_list = []

            for d in range(vol.shape[0]):
                image = vol[d]
                pad = 3
                padded = np.pad(image, pad, mode='constant', constant_values=0)

                if method == 'fast':
                    # very fast but probably not sufficient enough.
                    glcm = greycomatrix(image, [1], [0], levels=256, symmetric = True, normed = True)
                    # Extract all values for the whole image.
                    cont_value = greycoprops(glcm, prop='contrast').item()
                    corr_value = greycoprops(glcm, prop='correlation').item()
                    homo_value = greycoprops(glcm, prop='homogeneity').item()

                    print(cont_value, ' ', corr_value, ' ', homo_value)

                    corr_value_list.append(corr_value)

                elif method == 'sliding_window':
                    # Very slow and not efficient. (kind of the first try)
                    for row in range(image.shape[0]):
                        for column in range(image.shape[1]):
                            if image[row][column] == 0:
                                continue
                            if (padded.shape[0] - 7) == image.shape[0]:
                                break
                            glcm_window = padded[row:(row + 7), column:(column + 7)]
                            if np.any(glcm_window == 0):
                                continue

                            glcm = greycomatrix(glcm_window, [1], [0], levels=256, symmetric = False, normed = False)
                            corr_value = greycoprops(glcm, prop='correlation').item()
                            corr_value_list.append(corr_value)

                elif method == 'slic':
                    # best method to stick to.
                    n_seg = 10
                    offset = 1

                    mask = np.zeros(shape=image.shape, dtype=np.uint16)
                    mask[image > 0] = 1

                    slic_res = slic(image, n_segments=n_seg, compactness=0.01, mask=mask, start_label=1)

                    for s in range(np.amax(slic_res)):
                        index_list = np.argwhere(slic_res == s + 1).T
                        bbox = np.amin(index_list[0]) + offset, np.amax(index_list[0]) - offset, np.amin(index_list[1]) + offset, np.amax(index_list[1]) - offset #BBOX: rmin, rmax, cmin, cmax

                        tmpimg = image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                        if tmpimg.size == 0:
                            continue

                        glcm = greycomatrix(tmpimg, [1], [0], levels=256, symmetric = True, normed = True)
                        corr_value = greycoprops(glcm, prop='correlation').item()
                        corr_value_list.append(corr_value)
                else:
                    raise ValueError('No method defined. Please enter \'fast\' or \'sliding_window\'.')

            if idx % 50 == 0:
                print('Number of segments analysed: ', idx)

            texts[idx + 1] = np.array(corr_value_list)
    else:
        raise ValueError('Please enter valid dimensionality mode like 2d || 3d.')

    return (texts)


### HELPER SECTION ###
def rolling_window(a, window, step_size):
    '''
    Create a function to reshape a ndarray using a sliding window.
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def compute_bbox(image):
    '''
    Compute smallest boundingbox within an object.
    :param image: (np.array) 2d image
    '''
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (rmin, rmax, cmin, cmax)

def compute_bbox(image, label):
    '''
    Compute smallest boundingbox within an object.
    :param image: (np.array) 2d image
    :param label: (int)
    '''
    a = np.where(image == label)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox
