import numpy as np
import imageio
import glob
from skimage.measure import label, regionprops
from analyzer.data.data_vis import visvol
from scipy.ndimage.measurements import center_of_mass
from sklearn.metrics import normalized_mutual_info_score, pair_confusion_matrix

def clusteravg(values, labels):
    '''
    This function computes the average values of the associated cluster.
    :param values: (np.array) values (e.g. the area size) that are clustered.
    :param labels: (np.array) labels that where created by the clustering algorithm.

    :returns means: (list) the mean of the values contain a cluster.
    '''
    means = []
    for i in range(np.amax(labels) + 1):
        sums = np.sum(values, where=(labels == i))
        cnts = np.count_nonzero(labels == i)

        means.append(sums / cnts)

    return means

def cluster_res_single(imgn, gtfp):
    '''
    This function takes two images and compares them on the clustering result.
    :param imgn: (str) filename that is the result.
    :param gtn: (str) filename that is the groundtruth.
    '''
    gt = register_folder(gtfp, ff='png')
    img = imageio.imread(imgn)
    em = imageio.imread('datasets/human/human_em_export_8nm/human_em_export_s0220.png')
    #visvol(em, gt)
    #visvol(em, gt, filename='teshf_3_em_220', ff='png', save=True, dpi=1200)
    labels_pred = list()
    labels_true = list()
    labels, num_label = label(img, return_num=True)
    tmp = np.zeros(shape=img.shape, dtype=np.uint16)
    result = np.zeros(shape=(2,), dtype=np.uint16)
    for label_val in list(np.unique(labels)):
        if label_val == 0:
            continue
        new = np.where(labels == label_val, label_val, tmp)
        if np.unique(np.where(new == label_val, gt, tmp)).size == 1:
            labels_true.append(np.unique(np.where(new == label_val, gt, tmp))[0])
        else:
            labels_true.append(np.unique(np.where(new == label_val, gt, tmp))[1])
        labels_pred.append(np.unique(np.where(new == label_val, img, tmp))[1])

    print(labels_pred)
    _, counts_pred = np.unique(labels_pred, return_counts=True)
    print('preds: ', counts_pred)
    print(labels_true)
    _, counts_true = np.unique(labels_true, return_counts=True)
    print('true: ', counts_true)

    test = normalized_mutual_info_score(labels_true, labels_pred)
    #test = pair_confusion_matrix(labels_true, labels_pred)
    print(test)


### Helper section
def register_folder(fpath, ff='png'):
    '''fpath is the folderpath'''
    fns = sorted(glob.glob(fpath + '*.' + ff))
    for idx, fn in enumerate(fns):
        img = np.array(imageio.imread(fn), dtype=np.uint16)
        if idx == 0:
            result = img.copy().astype(int)
        else:
            result = np.add(result, img)
    for idx, label in enumerate(list(np.unique(result))):
        if idx == 0:
            continue
        else:
            result = np.where(result == label, idx, result)
    return result
