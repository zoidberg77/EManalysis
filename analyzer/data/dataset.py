import glob
import json
import math
import multiprocessing
import os
import re
import time

import h5py
import imageio
import numpy as np
import pandas as pd
from numpyencoder import NumpyEncoder
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.cluster import KMeans
from tqdm import tqdm

from analyzer.data.utils.data_raw import readvol, folder2Vol
from analyzer.utils.eval_model import Evaluationmodel


class Dataloader():
    '''
    Dataloader class for handling the em dataset and the related labels.

    :param cfg: configuration manager.
    :param volume: the EM volume.
    :param labels: labels that are defined by human or segmentation and will be clustered soon.
    :param gt: groundtruth data (cluster)
    :param feature: Defines the feature that the VAE should go for.
    :param chunk_size: (tuple) defines the chunks in which the data is loaded. Can help to overcome Memory errors.
    :param ff: (string) defines the file format that you want to work with. (default: png)
    '''

    def __init__(self, cfg, volume=None, labels=None, gt=None, feature="shape"):
        self.cfg = cfg
        if volume is not None:
            print('em data loaded: ', self.volume.shape)
        else:
            self.volpath = self.cfg.DATASET.EM_PATH
            self.volume = volume

        if labels is not None:
            print('label data loaded: ', self.labels.shape)
        else:
            self.labelpath = self.cfg.DATASET.LABEL_PATH
            self.labels = labels

        if gt is not None:
            print('gt data loaded: ', self.gt.shape)
        else:
            self.gtpath = self.cfg.DATASET.GT_PATH
            self.gt = gt

        self.chunk_size = self.cfg.DATASET.CHUNK_SIZE
        self.ff = self.cfg.DATASET.FILE_FORMAT
        if self.cfg.SYSTEM.NUM_CPUS is None:
            self.cpus = multiprocessing.cpu_count()
        else:
            self.cpus = self.cfg.SYSTEM.NUM_CPUS

        self.region_limit = cfg.AUTOENCODER.REGION_LIMIT
        self.chunks_per_cpu = cfg.AUTOENCODER.CHUNKS_CPU
        self.upper_limit = cfg.AUTOENCODER.UPPER_BOUND
        self.lower_limit = cfg.AUTOENCODER.LOWER_BOUND
        self.large_samples = cfg.AUTOENCODER.LARGE_OBJECT_SAMPLES
        self.target_size = cfg.AUTOENCODER.TARGET
        self.vae_feature = feature
        self.mito_volume_file_name = "{}mito_samples.h5".format(cfg.DATASET.ROOTD)
        self.exclude_borders = cfg.DATASET.EXCLUDE_BORDER_OBJECTS

    def __len__(self):
        '''
        Required by torch to return the length of the dataset.
        :returns: integer
        '''
        with h5py.File(self.mito_volume_file_name, 'r') as f:
            return len(f["id"])

    def __getitem__(self, idx):
        '''
        Required by torch to return one item of the dataset.
        :param idx: index of the object
        :returns: object from the volume
        '''
        with h5py.File(self.mito_volume_file_name, 'r') as f:
            return f["chunk"][idx], idx

    def get_fns(self):
        '''returns the em, label and gt filenames of every image.'''
        emfns = sorted(glob.glob(self.volpath + '*.' + self.ff))
        labelfns = sorted(glob.glob(self.labelpath + '*.' + self.ff))
        gtfns = sorted(glob.glob(self.gtpath + '*.' + self.ff))
        return (emfns, labelfns, gtfns)

    def load_chunk(self, vol='all', mode='3d'):
        '''
        Load chunk of em and groundtruth data for further processing.
        :param vol: (string) choose between -> 'all', 'em', 'label' in order to specify
                     with volume you want to load.
        '''
        emfns = sorted(glob.glob(self.volpath + '*.' + self.ff))
        labelfns = sorted(glob.glob(self.labelpath + '*.' + self.ff))
        if mode == '2d':
            if (vol == 'em') or (vol == 'all'):
                emdata = readvol(emfns[0])
                emdata = np.squeeze(emdata)
                print('em data loaded: ', emdata.shape)
            if (vol == 'label') or (vol == 'all'):
                labels = readvol(labelfns[0])
                labels = np.squeeze(labels)
                print('label data loaded: ', labels.shape)

        if mode == '3d':
            if (vol == 'em') or (vol == 'all'):
                if self.volume is None:
                    emdata = folder2Vol(self.volpath, self.chunk_size, file_format=self.ff)
                    print('em data loaded: ', emdata.shape)
            if (vol == 'label') or (vol == 'all'):
                if self.labels is None:
                    labels = folder2Vol(self.labelpath, self.chunk_size, file_format=self.ff)
                    print('label data loaded: ', labels.shape)
            if (vol == 'gt') or (vol == 'all'):
                if self.gt is None:
                    gt = folder2Vol(self.gtpath, self.chunk_size, file_format=self.ff)
                    print('gt data loaded: ', gt.shape)

        return (emdata, labels, gt)

    def list_segments(self, vol, labels, min_size=2000, os=0, mode='3d'):
        '''
        This function creats a list of arrays that contain the unique segments.
        :param vol: (np.array) volume that contains the pure em data. (2d || 3d)
        :param label: (np.array) volume that contains the groundtruth. (2d || 3d)
        :param min_size: (int) this sets the minimum size of mitochondria region in order to be safed to the list. Used only in 2d.
        :param os: (int) defines the offset that should be used for cutting the bounding box. Be careful with offset as it can lead to additional regions in the chunks.
        :param mode: (string) 2d || 3d --> 2d gives you 2d arrays of each slice (same mitochondria are treated differently as they loose their touch after slicing)
                                       --> 3d gives you the whole mitochondria in a 3d volume.

        :returns: (dict) of (np.array) objects that contain the segments with labels as keys.
        '''
        bbox_dict = {}
        mask = np.zeros(shape=vol.shape, dtype=np.uint16)
        mask[labels > 0] = 1
        vol[mask == 0] = 0

        if mode == '2d':
            bbox_list = []
            for idx in range(vol.shape[0]):
                image = vol[idx, :, :]
                gt_img = labels[idx, :, :]
                label2d, num_label = label(gt_img, return_num=True)
                regions = regionprops(label2d, cache=False)

                for props in regions:
                    boundbox = props.bbox
                    if props.bbox_area > min_size:
                        if ((boundbox[0] - os) < 0) or ((boundbox[2] + os) > image.shape[0]) or (
                                (boundbox[1] - os) < 0) or ((boundbox[3] + os) > image.shape[1]):
                            tmparr = image[boundbox[0]:boundbox[2], boundbox[1]:boundbox[3]]
                        else:
                            tmparr = image[(boundbox[0] - os):(boundbox[2] + os), (boundbox[1] - os):(boundbox[3] + os)]
                        bbox_list.append(tmparr)

            bbox_dict = {i: bbox_list[i] for i in range(len(bbox_list))}

        elif mode == '3d':
            chunk_dict = {}

            label3d, num_label = label(labels, return_num=True)
            regions = regionprops(label3d, cache=False)

            for props in regions:
                boundbox = props.bbox
                if ((boundbox[1] - os) < 0) or ((boundbox[4] + os) > vol.shape[1]) or ((boundbox[2] - os) < 0) or (
                        (boundbox[5] + os) > vol.shape[2]):
                    tmparr = vol[boundbox[0]:boundbox[3], boundbox[1]:boundbox[4], boundbox[2]:boundbox[5]]
                else:
                    tmparr = vol[boundbox[0]:boundbox[3], (boundbox[1] - os):(boundbox[4] + os),
                             (boundbox[2] - os):(boundbox[5] + os)]
                bbox_dict[props.label] = tmparr
        else:
            raise ValueError('No valid dimensionality mode in function list_segments.')

        return (bbox_dict)

    def prep_data_info(self, volopt='label', save=False):
        '''
        This function aims as an inbetween function iterating over the whole dataset in efficient
        and memory proof fashion in order to preserve information that is needed for further steps.
        :param volopt: (string) this sets the volume you want to use for the operation. default: gt
        :param kernel_n: (int) number of CPU kernels you want to use for multiprocessing.

        :returns added: (dict) that contains the labels with respective information as (list): [pixelsize, [slice_index(s)]]
        '''
        if volopt == 'label':
            fns = sorted(glob.glob(self.labelpath + '*.' + self.ff))
        elif volopt == 'em':
            fns = sorted(glob.glob(self.volpath + '*.' + self.ff))
        else:
            raise ValueError('Please enter the volume on which \'prep_data_info\' should run on.')

        if self.exclude_borders:
            fns = fns[1:-1]

        with multiprocessing.Pool(processes=self.cpus) as pool:
            result = pool.starmap(self.calc_props, enumerate(fns))

        added = {}
        for dicts in result:
            for key, value in dicts.items():
                if key in added:
                    added[key][0] += value[0]
                    added[key][1].append(value[1])
                    if self.exclude_borders:
                        if not added[key][2]:
                            added[key].append(value[2])

                else:
                    added.setdefault(key, [])
                    added[key].append(value[0])
                    added[key].append([value[1]])
                    if self.exclude_borders:
                        added[key].append(value[2])

        result_array = []
        for result in added.keys():
            if self.exclude_borders and added[result][2]:
                continue
            result_array.append({
                'id': result,
                'size': added[result][0],
                'slices': added[result][1]
            })
        if save:
            with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'w') as f:
                json.dump(result_array, f, cls=NumpyEncoder)
                f.close()

        return (result_array)

    def calc_props(self, idx, fns):
        '''
        Helper function for 'prep_data_info'
        :param idx: (int) this is the slice index that correspondes to the image slice. E.g. idx 100 belongs to image 100.
        :param fns: (string) list of filenames.
        :returns result: (dict) with each segment. key: idx of segment -- value: [number of pixels in segment, idx of slice].
        '''
        result = {}
        if os.path.exists(fns):
            tmp = imageio.imread(fns)
            for region in regionprops(tmp):
                result.setdefault(region.label, [])
                result[region.label].append(region.area)
                result[region.label].append(idx)
                result[region.label].append(False)
                if self.exclude_borders:
                    minr, minc, maxr, maxc = region.bbox
                    if minr == 0 or minc == 0:
                        result[region.label][-1] = True
                    if maxr == tmp.shape[0] or maxc == tmp.shape[0]:
                        result[region.label][-1] = True

        return result

    def precluster(self, mchn='simple', n_groups=5):
        '''
        Function preclusters the mitochondria into buckets of similar size in order to avoid
        sparsity and loss of information while extracting latent representation of the mitochondria.
        '''
        if os.path.exists(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)) \
                and os.stat(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)).st_size != 0:
            with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'r') as f:
                data_info = json.loads(f.read())
        else:
            data_info = self.prep_data_info(save=True)

        tmp = np.stack(([mito['id'] for mito in data_info], [mito['size'] for mito in data_info]), axis=-1)

        if mchn == 'simple':
            sorted = tmp[tmp[:, 1].argsort()[::-1]]
            splitted = np.array_split(sorted, n_groups, axis=0)
            id_lists = [tmp[:, 0].tolist() for tmp in splitted]

        elif mchn == 'cluster':
            model = KMeans(n_clusters=n_groups)
            res_grps = model.fit_predict(np.array(tmp[:, 1]).reshape(-1, 1))
            id_lists = [[]] * n_groups
            for idx in range(len(res_grps)):
                id_lists[res_grps[idx]].append(tmp[:, 0][idx])
        else:
            raise ValueError(
                'Please enter the a valid mechanismn you want to group that mitochondria. \'simple\' or \'cluster\'.')

        return id_lists

    def extract_scale_mitos(self):
        '''
        Function to extract the objects as volumes and scale them. Then its saves the scaled volumes to an h5 file.
        '''
        if os.path.exists(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)) \
                and os.stat(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO)).st_size != 0:
            with open(os.path.join(self.cfg.SYSTEM.ROOT_DIR, self.cfg.DATASET.DATAINFO), 'r') as f:
                regions = json.loads(f.read())
        else:
            regions = self.prep_data_info(save=False)

        print("{} objects found in the ground truth".format(len(regions)))

        regions = pd.DataFrame(regions)
        regions = regions[(self.upper_limit > regions['size']) & (self.lower_limit < regions['size']) & (
                len(regions['slices']) > 1)].values.tolist()
        filtered_length = len(regions)
        print("{} within limits {} and {}".format(filtered_length, self.lower_limit, self.upper_limit))
        if self.region_limit is not None:
            regions = regions[:self.region_limit]
            print("{} will be extracted due to set region_limit".format(self.region_limit))
        with h5py.File(self.mito_volume_file_name, "w") as f:
            f.create_dataset("shape_volume", (len(regions), 1, *self.target_size))
            f.create_dataset("texture_volume", (len(regions), 1, *self.target_size))
            f.create_dataset("id", (len(regions),))

            if self.cpus < 2 and self.chunks_per_cpu < 2:
                print("single cpu mode")
                for i in tqdm(range(0, len(regions))):
                    if i < 11000:
                        continue
                    print(i)
                    result = self.get_mito_volume(regions[i])
                    f["id"][i] = result[0]
                    f["shape_volume"][i] = result[1]
                    f["texture_volume"][i] = result[2]

            with multiprocessing.Pool(processes=self.cpus) as pool:
                for i in tqdm(range(0, len(regions), int(self.cpus * self.chunks_per_cpu))):
                    try:
                        results = pool.map(self.get_mito_volume, regions[i:i + int(self.cpus * self.chunks_per_cpu)])
                        for j, result in enumerate(results):
                            f["id"][i + j] = result[0]
                            f["shape_volume"][i + j] = result[1]
                            f["texture_volume"][i + j] = result[2]
                    except:
                        print("error in extraction, i: {}".format(i))
                        exit()

    def get_mito_volume(self, region):
        '''
        Preprocessing function to extract and scale the mitochondria as volume
        :param region: (dict) one region object provided by Dataloader.prep_data_info
        :returns result: (numpy.array) a numpy array with the target dimensions and the mitochondria in it
        '''
        gt_volume, em_volume = self.get_volumes_from_slices(region)

        mito_regions = regionprops(gt_volume, cache=False)
        if len(mito_regions) != 1:
            print("something went wrong during volume building. region count: {}".format(len(mito_regions)))

        mito_region = mito_regions[0]

        if len(mito_region.bbox) < 6:
            return [-1, np.zeros(shape=(1, *self.target_size)), np.zeros(shape=(1, *self.target_size))]

        shape = gt_volume[mito_region.bbox[0]:mito_region.bbox[3] + 1,
                mito_region.bbox[1]:mito_region.bbox[4] + 1,
                mito_region.bbox[2]:mito_region.bbox[5] + 1].astype(np.float32)

        texture = em_volume[mito_region.bbox[0]:mito_region.bbox[3] + 1,
                  mito_region.bbox[1]:mito_region.bbox[4] + 1,
                  mito_region.bbox[2]:mito_region.bbox[5] + 1].astype(np.float32)

        scaled_shape = resize(shape, self.target_size, order=1, anti_aliasing=True)
        scaled_shape = scaled_shape / scaled_shape.max()
        scaled_shape = np.expand_dims(scaled_shape, 0)

        scaled_texture = resize(texture, self.target_size, order=1, anti_aliasing=True)
        scaled_texture = scaled_texture / scaled_texture.max()
        scaled_texture = np.expand_dims(scaled_texture, 0)
        if scaled_shape.sum() < self.lower_limit * 0.1:
            print("region {} was too small".format(region[0]))
            return [-1, np.zeros(shape=(1, *self.target_size)), np.zeros(shape=(1, *self.target_size))]

        return [region[0], scaled_shape, scaled_texture]

    def get_volumes_from_slices(self, region):
        '''
        #TODO
        :param region:
        :returns gt_volume, em_volume:
        '''
        gt_all_fn = sorted(glob.glob(self.labelpath + '*.' + self.ff))
        em_all_fn = sorted(glob.glob(self.volpath + '*.' + self.ff))

        gt_fns = [gt_all_fn[id] for id in region["slices"]]
        em_fns = [em_all_fn[id] for id in region["slices"]]

        gt_volume = []
        em_volume = []

        for i in range(len(gt_fns)):
            gt_slice = imageio.imread(gt_fns[i])
            em_slice = imageio.imread(em_fns[i])

            gt_slice[gt_slice != region["id"]] = 0
            em_slice[gt_slice != region["id"]] = 0

            gt_volume.append(gt_slice)
            em_volume.append(em_slice)

        return np.array(gt_volume), np.array(em_volume)

    def extract_scale_mitos_samples(self):
        '''
        Function to extract the objects as volumes and scale them. Then its saves the scaled volumes to an h5 file.
        '''
        regions = self.prep_data_info(save=True)

        print("{} objects found in the ground truth".format(len(regions)))

        regex = re.compile('([0-9]+)_mito_samples.h5')
        for root, dirs, files in os.walk(self.cfg.DATASET.ROOTD):
            for file in files:
                if regex.match(file):
                    os.remove(self.cfg.DATASET.ROOTD + file)

        in_q = multiprocessing.Queue()
        processes = []

        for region in regions:
            in_q.put(region)
        pbar = tqdm(total=len(regions))
        for cpu in range(self.cpus):
            p = multiprocessing.Process(target=self.get_mito_chunk, args=(in_q, cpu))
            p.start()
            processes.append(p)

        progress = 0
        while not in_q.empty():
            progress = len(regions)-in_q.qsize()-progress
            time.sleep(30)
            pbar.update(progress)
        for p in processes:
            p.join()

        self.cleanup_h5()
        return

    def get_mito_chunk(self, in_q, id):
        h5file = self.cfg.DATASET.ROOTD + "{}_mito_samples.h5".format(id)
        with h5py.File(h5file, "w") as f:
            counter = 0
            chunks = f.create_dataset("chunk", (1, 1,  *self.target_size),
                                      maxshape=(None, 1, *self.target_size))
            ids = f.create_dataset("id", (1,), maxshape=(None,))

            while True:
                if in_q.empty():
                    break
                region = in_q.get(timeout=10)
                gt_volume, em_volume = self.get_volumes_from_slices(region)

                mito_regions = regionprops(gt_volume, cache=False)
                if len(mito_regions) != 1:
                    continue
                mito_region = mito_regions[0]
                texture = None

                if len(mito_region.bbox) < 6:
                    texture = np.zeros((*em_volume.shape, 2))
                    texture[mito_region.bbox[0]:mito_region.bbox[2] + 1,
                    mito_region.bbox[1]:mito_region.bbox[3] + 1, 0] = em_volume[
                                                                      mito_region.bbox[0]:mito_region.bbox[2] + 1,
                                                                      mito_region.bbox[1]:mito_region.bbox[3] + 1]
                else:
                    texture = em_volume[mito_region.bbox[0]:mito_region.bbox[3] + 1,
                              mito_region.bbox[1]:mito_region.bbox[4] + 1,
                              mito_region.bbox[2]:mito_region.bbox[5] + 1].astype(np.float32)

                large = any([d > self.target_size[i] for i, d in enumerate(texture.shape)])

                if large:
                    for i in range(self.large_samples):

                        x, y, z = 0, 0, 0

                        if texture.shape[0] > self.target_size[0]:
                            z = np.random.random_integers(0, texture.shape[0] - self.target_size[0])
                        if texture.shape[1] > self.target_size[1]:
                            x = np.random.random_integers(0, texture.shape[1] - self.target_size[1])
                        if texture.shape[2] > self.target_size[2]:
                            y = np.random.random_integers(0, texture.shape[2] - self.target_size[2])

                        sample = texture[
                                 z:z + self.target_size[0],
                                 x:x + self.target_size[1],
                                 y:y + self.target_size[2]]
                        if np.count_nonzero(sample) / sample.size < 0.1:
                            continue
                        sample_padding = np.zeros(self.target_size)

                        sample_padding[0:texture.shape[0], 0:texture.shape[1], 0:texture.shape[2]] = sample
                        if len(chunks) <= counter:
                            chunks.resize((chunks.shape[0] + 1), axis=0)
                            ids.resize((ids.shape[0] + 1), axis=0)
                        np.expand_dims(sample_padding, 0)
                        chunks[counter] = sample_padding
                        ids[counter] = region["id"]
                        counter += 1

                else:
                    sample_padding = np.zeros(self.target_size)
                    sample_padding[0:texture.shape[0], 0:texture.shape[1], 0:texture.shape[2]] = texture
                    if len(chunks) <= counter:
                        chunks.resize((chunks.shape[0] + 1), axis=0)
                        ids.resize((ids.shape[0] + 1), axis=0)
                    np.expand_dims(sample_padding, 0)
                    chunks[counter] = sample_padding
                    ids[counter] = region["id"]
                    counter += 1

        return

    def cleanup_h5(self):
        eval_model = Evaluationmodel(cfg=self.cfg, dl=self)

        regex = re.compile('([0-9]+)_mito_samples.h5')
        size_needed = 0
        for root, dirs, files in os.walk(self.cfg.DATASET.ROOTD):
            for file in files:
                if regex.match(file):
                    with h5py.File(self.cfg.DATASET.ROOTD + file, "r") as f:
                        size_needed += len(f["id"])

        counter = 0
        with h5py.File(self.cfg.DATASET.ROOTD + "mito_samples.h5", "w") as mainf:
            chunks = mainf.create_dataset("chunk", (size_needed, 1, *self.target_size))
            ids = mainf.create_dataset("id", (size_needed,))
            gts = mainf.create_dataset("gt", (size_needed,))

            for root, dirs, files in os.walk(self.cfg.DATASET.ROOTD):
                for file in files:
                    if regex.match(file):
                        with h5py.File(self.cfg.DATASET.ROOTD + file, "r") as f:
                            for id, chunk in zip(f["id"], f["chunk"]):
                                ids[counter] = id
                                chunks[counter] = chunk
                                counter += 1
                        os.remove(self.cfg.DATASET.ROOTD + file)

            gt_vector = eval_model.fast_create_gt_vector(save=False)
            gt_lookup = {}
            sort_ids = sorted(list(ids))
            for i, id in enumerate(sort_ids):
                gt_lookup[str(id)] = gt_vector[i]

            for i, id in enumerate(ids):
                gts[i] = gt_lookup[str(id)]

            print("samples collected: {}".format(len(mainf["id"])))
