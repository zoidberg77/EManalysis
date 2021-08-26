import os, sys
import h5py
import numpy as np

# with h5py.File(os.path.join('datasets/vae/mouseA/pts.h5'), 'r') as h5f:
#     ptcs_dict = h5f.get('ptcs')
#
#     int_keys = np.array([int(x) for x in ptcs_dict.keys()])
#     labels = np.array(h5f["labels"][:])
#     print(ptcs_dict)
#     print(int_keys)
#     print(np.array(h5f["labels"][:]))
#
#     for idx in range(labels.shape[0]):
#         elem = labels[idx]
#         if elem in int_keys:
#             continue
#         else:
#             raiseValueError('Error')

with h5py.File(os.path.join('models/ptc/human/rec_pts.h5'), 'r') as h5f:
    print("Keys: %s" % h5f.keys())
    grp = list(h5f.keys())[0]
    print(grp)

    # Get the data
    data = list(h5f[grp])
    print(data)
