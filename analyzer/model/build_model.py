import os, sys
import numpy as np
import h5py
import imageio
#import hdbscan
from scipy.spatial import distance
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analyzer.model.utils.helper import *
from analyzer.data.data_vis import visvol, vissegments
from analyzer.utils.eval_model import Evaluationmodel

from .feat_extr_model import FeatureExtractor

class Clustermodel():
    '''
    Setups up the model for running a clustering algoritm on the loaded data.
    :param cfg: configuration management. This sets basically all the parameters.
    :param emvol & gtvol: (np.array) Both are the data volumes.
    :param dl: (class object) This is the dataloader class object.
    :param alg: sets the clustering algorithm that should be used. (default: KMeans)
                - 'kmeans': KMeans
                - 'affprop': AffinityPropagation
                - 'specCl': SpectralClustering
                - 'aggloCl': AgglomerativeClustering
                - 'dbscan': DBSCAN
                - 'hdbscan': HDBSCAN (https://hdbscan.readthedocs.io/en/latest/index.html)

    :param n_cluster: (int) sets the number of cluster that should be found.
    :param feat_list: ['sizef', 'distf', 'shapef', 'textf', 'circf'] -- choose from different features you want to use for clustering.
    :param weightsf: [1, 1, 1 ,1, 1] -- weight each individual feature and therefore their influence on the clustering.
    '''
    def __init__(self, cfg, emvol=None, gtvol=None, dl=None):
        self.cfg = cfg
        self.emvol = emvol
        self.gtvol = gtvol
        self.dl = dl
        self.alg = self.cfg.CLUSTER.ALG
        self.feat_list = self.cfg.CLUSTER.FEAT_LIST
        self.weightsf = self.cfg.CLUSTER.WEIGHTSF
        self.n_cluster = self.cfg.CLUSTER.N_CLUSTER

        self.model = self.set_model(mn=self.alg)
        self.fe = FeatureExtractor(self.cfg)
        self.eval = Evaluationmodel(self.cfg, self.dl)

        print(' --- model is set. algorithm: {}, clustering by the features: {} --- '.format(self.alg, str(self.feat_list).strip('[]')))

    def set_model(self, mn='kmeans'):
        '''
        This function enables the usage of different algoritms when setting the model overall.
        :param mn: (string) that is the name of the algoritm to go with.
        '''
        if mn == 'kmeans':
            model = KMeans(n_clusters=self.n_cluster)
        elif mn == 'affprop':
            model = AffinityPropagation()
        elif mn == 'specCl':
            model = SpectralClustering(n_clusters=self.n_cluster)
        elif mn == 'dbscan':
            model = DBSCAN(eps=0.05, n_jobs=-1)
        elif mn == 'hdbscan':
            model = hdbscan.HDBSCAN(min_cluster_size=self.n_cluster, min_samples=500, gen_min_span_tree=True)
        elif mn == 'aggloCl':
            model = AgglomerativeClustering(n_clusters=self.n_cluster, affinity='precomputed', linkage='single')
        else:
            raise ValueError('Please enter a valid clustering algorithm. -- \'kmeans\', \'affprop\', \'specCl\', \'dbscan\', \'hdbscan\', \'aggloCl\'')

        return model

    def get_features(self):
        '''
        This function will load different features vectors that were extracted and saved to be used for clustering.
        :param feat_list: (list) of (string)s that states which features should be computed and/or load to cache for
                            further processing.
        :returns labels: (np.array) that contains the labels.
        :returns rs_feat_list: (list) of (np.array)s that contain the related features.
        '''
        rs_feat_list = list()
        labels = np.array([])
        for idx, fns in enumerate(self.feat_list):
            if os.path.exists(self.cfg.DATASET.ROOTF + fns + '.h5') is False:
                print('This file {} does not exist, will be computed.'.format(self.cfg.DATASET.ROOTF + fns + '.h5'))

                if fns == 'sizef':
                    feat = self.fe.compute_seg_size()
                elif fns == 'distf':
                    feat = self.fe.compute_seg_dist()
                elif fns == 'shapef':
                    feat = self.fe.compute_vae_shape()
                elif fns == 'ptc_shapef':
                    feat = self.fe.compute_vae_ptc_shape()
                elif fns == 'texturef':
                    feat = self.fe.compute_vae_texture()
                elif fns == 'circf':
                    feat = self.fe.compute_seg_circ()
                elif fns == 'surface_to_volumef':
                    feat = self.fe.compute_seg_surface_to_volume()
                elif fns == 'slenf':
                    feat = self.fe.compute_seg_slength()
                elif fns == 'spatial_densityf':
                    volume, count = self.fe.compute_seg_spatial_density(n_block=30)
                    with h5py.File(self.cfg.DATASET.ROOTF + "spatial_densityf" + '.h5', 'w') as h5f:
                        h5f.create_dataset('volume', data=volume)
                        h5f.create_dataset('count', data=count)
                    exit()
                else:
                    print('No function for computing {} features.'.format(fns))
                    raise ValueError('Please check {} if it is correct.'.format(fns))

                label, values = self.fe.save_single_feat_h5(feat, filen=fns)
                if labels.size == 0:
                    labels = np.array(label, dtype=np.uint16)
                rs_feat_list.append(np.array(values))
            else:
                fn = self.cfg.DATASET.ROOTF + fns + '.h5'
                with h5py.File(fn, "r") as h5f:
                    if labels.size == 0:
                        labels = np.array(h5f['id'], dtype=np.uint16)

                    rs_feat_list.append(np.array(h5f[fns[:-1]]))
                    print('Loaded {} features to cache.'.format(fns[:-1]))
                    test = np.array(h5f[fns[:-1]])
                    print('\nfeature vector {} has shape {}'.format(fn, test.shape))

            if idx == 0:
                base_labels = labels
            else:
                if check_feature_order(base_labels, labels) is False:
                    print('ORDER IS WRONG. Correct the order of {} features.'.format(fns))
                    ordered_feat = correct_idx_feat(base_labels, labels, rs_feat_list[idx])
                    rs_feat_list[idx] = ordered_feat

        return labels, rs_feat_list

    def prep_cluster_matrix(self, labels, feat_list, load=False, save=False):
        '''
        Function computes clustering matrix from different features for the actual clustering.
        :param labels:
        :param feat_list: (list) of (np.array)s that are the feature vectors/matrices.
        :param weights: (np.array) of weighting factor for different features.
        :returns clst_m: (np.array) of NxN clustering distance from each feature to another. N is a sample.
        '''
        #Preload if possible.
        if load and os.path.exists(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')) \
                and os.stat(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5')).st_size != 0:
            print('preload the clustering matrix.')
            with h5py.File(os.path.join(self.cfg.DATASET.ROOTF, 'clstm.h5'), "r") as h5f:
                clst_m = np.array(h5f['clstm'])
                h5f.close()
        else:
            print('computing the clustering matrix.')
            scaler = MinMaxScaler()
            clst_m = np.zeros(shape=feat_list[0].shape[0], dtype=np.float16)
            for idx, feat in enumerate(feat_list):
                if feat.ndim <= 1:
                    tmp = scaler.fit_transform(feat.reshape(-1,1))
                    clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * distance.cdist(tmp, tmp, 'euclidean'))
                else:
                    if feat.shape[0] == feat.shape[1]:
                        clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * min_max_scale(feat))
                    else:
                        tmp = min_max_scale(feat)
                        clst_m = np.add(clst_m, self.cfg.CLUSTER.WEIGHTSF[idx] * distance.cdist(tmp, tmp, 'euclidean'))

            clst_m = np.vstack(clst_m)
            if save == True:
                self.fe.save_feats_h5(labels, clst_m, filen='clstm')
        return clst_m

    def run(self):
        '''
        Running the main clustering algoritm on the features (feature list) extracted.
        '''
        labels, feat = self.get_features()
        clst_m = self.prep_cluster_matrix(labels, feat)
        res_labels = self.model.fit_predict(clst_m)
        self.eval.eval(res_labels)

        if self.cfg.CLUSTER.GENERATE_MASKS:
            _, gtfns = self.fe.get_fns()
            _ = recompute_from_res(labels, res_labels, volfns=gtfns, dprc=self.cfg.MODE.DPRC, fp=self.cfg.CLUSTER.OUTPUTPATH + "masks/", neuroglancer=self.cfg.CLUSTER.NEUROGLANCER, em_path=self.cfg.DATASET.EM_PATH)
            self.eval.eval_volume(res_labels)

        if self.cfg.CLUSTER.VISUALIZATION:
            # For visualization purposes.
            em_files = glob.glob(self.cfg.DATASET.EM_PATH + '*.' + self.cfg.DATASET.FILE_FORMAT)
            labeled_files = glob.glob(self.cfg.CLUSTER.OUTPUTPATH + 'masks/*.' + self.cfg.DATASET.FILE_FORMAT)

            for idx, em_file in enumerate(em_files):
                labeled = imageio.imread(labeled_files[idx])
                em = imageio.imread(em_file)
                visvol(em, labeled, filename=(self.cfg.CLUSTER.OUTPUTPATH + "overlay/{}".format(idx)), save=True)

        print('\nfinished clustering.')
