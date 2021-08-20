import os, sys
import numpy as np
import h5py
import json
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

parent = os.path.abspath(os.getcwd())
sys.path.append(parent)

from analyzer.model.utils.extracting import compute_region_size


mouse_cell_type_vector_all = ['myelinated axon', 'spiny dendrite', 'unsure dendrite', 'smooth dendrite', 'e_axon',
                        'i_axon', 'presynaptic bouton(ex)', 'presynaptic bouton(in)', 'dendritic spine (ex)',
                        'cellbodies', 'glia', 'unsure axon', 'unsure neurite']

mouse_cell_type_vector_main = ['cellbodies', 'glia', 'unsure', 'axons', 'dendrites']


with h5py.File(parent + '/features/mouseA/sizef.h5', "r") as h5f:
    mouse_labels = np.array(h5f['id'], dtype=np.uint16)
    sizef_mouse = np.array(h5f['size'])
    print('Loaded mouse {} features to cache.'.format('size'))

with h5py.File(parent + '/features/human/sizef.h5', "r") as h5f:
    human_labels = np.array(h5f['id'], dtype=np.uint16)
    sizef_human = np.array(h5f['size'])
    print('Loaded human {} features to cache.'.format('size'))

with h5py.File(parent + '/features/rat/sizef.h5', "r") as h5f:
    rat_labels = np.array(h5f['id'], dtype=np.uint16)
    sizef_rat = np.array(h5f['size'])
    print('Loaded rat {} features to cache.'.format('size'))

with open(os.path.join(parent, 'features/mouseA/gt_vector_allgroups.json'), 'r') as f:
    gt_vector_all = np.array(json.loads(f.read()))
    print('all cells: ', np.unique(gt_vector_all))

with open(os.path.join(parent, 'features/mouseA/gt_vector.json'), 'r') as f:
    gt_vector_main = np.array(json.loads(f.read()))
    gt_main_values, gt_main_counts = np.unique(gt_vector_main, return_counts=True)
    print('main cells: ', gt_main_values)
    print('count main cells: ', gt_main_counts)

all_cell_labels = list()
for i in range(gt_vector_all.shape[0]):
    label = gt_vector_all[i]
    idx = np.argwhere(np.unique(gt_vector_all) == gt_vector_all[i])[0][0]
    all_cell_labels.append(mouse_cell_type_vector_all[idx])

main_cell_labels = list()
for i in range(gt_vector_main.shape[0]):
    label = gt_vector_main[i]
    idx = np.argwhere(np.unique(gt_vector_main) == gt_vector_main[i])[0][0]
    main_cell_labels.append(mouse_cell_type_vector_main[idx])

vol_data = np.swapaxes(np.stack((sizef_mouse, gt_vector_main)), 0, 1)
mouse_dataset = pd.DataFrame({'label': mouse_labels, 'gt_label': gt_vector_main, 'volume [number of pixel]': sizef_mouse})
human_dataset = pd.DataFrame({'label': human_labels, 'volume [number of pixel]': sizef_human})
rat_dataset = pd.DataFrame({'label': rat_labels, 'volume [number of pixel]': sizef_rat})

print(mouse_dataset)

fig, ax = plt.subplots(1, 3, figsize=(5000, 5))
#fig.set_figheight(5)
#fig.suptitle('Size feature distribution over various datasets', fontsize=10)
sb.set_style('darkgrid')
sb.despine()
sb.set_context('paper', rc={"font.size":12, "axes.titlesize":12, "axes.labelsize":5})
ax[0].set_title('Human dataset')
sb.histplot(data=human_dataset, x='volume [number of pixel]', kde=True, log_scale=True, ax=ax[0], palette='Blues_d')
ax[1].set_title('Mouse dataset')
sb.histplot(data=mouse_dataset, x='volume [number of pixel]', kde=True, log_scale=True, ax=ax[1])
ax[2].set_title('Rat dataset')
sb.histplot(data=rat_dataset, x='volume [number of pixel]', kde=True, log_scale=True, ax=ax[2])

plt.show()
