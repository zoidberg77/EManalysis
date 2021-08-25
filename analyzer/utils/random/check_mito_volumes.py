import h5py
import sys
import numpy as np
from tqdm import tqdm

with h5py.File(sys.argv[1], 'r') as f:
    counter = 0
    for i, volume in enumerate(tqdm(f[sys.argv[2]])):
        if np.max(volume) != 0:
            counter += 1
    print("mitochondria saved: {}".format(counter))