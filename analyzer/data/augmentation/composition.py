import numpy as np
from skimage.filters import gaussian

class Compose(object):
    '''Composing a list of data transforms for handling 3d volumes.
    '''
    def __init__(self,
                 transforms: list = []):
        self.transforms = transforms

    def __call__(self, sample, random_state=np.random.RandomState()):
        ran = random_state.rand(len(self.transforms))
        for id, apply_trans in enumerate(reversed(self.transforms)):
            sample = apply_trans(sample, random_state)

        return sample
