import numpy as np

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
