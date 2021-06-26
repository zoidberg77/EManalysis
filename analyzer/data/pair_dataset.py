import numpy as np
import torch
import torch.utils.data

class PairDataset(VolumeDataset):
    '''
    This Dataloader will prepare sample that are pairs for feeding the contrastive
    learning algorithm.
    '''
    def __init__(self):
        self.num_augmented_images = 2

    def __len__(self):
        pass

    def __getitem__(self, idx):
        sample_pair = self._create_sample_pair()
        return sample_pair

    def create_sample_pair(self):
        '''Create a sample pair that will be used for contrastive learning.
        '''
        sample_pair = list()

        sample = self._random_sampling(self.sample_volume_size)
        pos, out_volume, out_label, out_valid = sample
        out_volume = self._create_masked_input(out_volume, out_label)

        data = {'image': out_volume}
        for i in range(self.num_augmented_images):
            augmented = self.augmentor(data)
            sample_pair.append(augmented['image'])

        return sample_pair

    def create_chunk_volume(self):
        '''
        Function creates small chunk from input volume that is processed
        into the training model.
        '''
        pass

    def create_masked_input(self, vol: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Create masked input volume, that is pure EM where the mask is not 0. Otherwise all
        values set to 0. Returns the prepared mask.
        :params vol (numpy.ndarray): volume that is EM input.
        :params label (numpy.ndarray): associated label volume.
        '''
        vol[np.where(label == 0)] = 0
        return vol
