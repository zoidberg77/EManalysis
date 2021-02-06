import h5py


class MitoDataset():
    def __init__(self, mito_h5_file, dataset_name):
        self.volume = h5py.File(mito_h5_file, 'r')
        self.dataset_name = dataset_name

    def __len__(self):
        '''
        Required by torch to return the length of the dataset.
        :returns: integer
        '''
        return self.volume[self.dataset_name].shape[0]

    def __getitem__(self, idx):
        '''
        Required by torch to return one item of the dataset.
        :param idx: index of the object
        :returns: object from the volume
        '''
        return self.volume[self.dataset_name][idx]