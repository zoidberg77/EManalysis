import os, sys
import numpy as np
import h5py
import json
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import re
from typing import Tuple, List, Optional
#
#   This file serves as an analysis board where model states should be visualized.
#
def read_log(txt_file, column_list=['epoch', 'iteration', 'loss', 'lr']):
    '''reading log/txt file. Returns pandas dataframe.'''
    df = pd.DataFrame(columns=column_list)
    with open(txt_file) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            cols = line.split()
            content = list()
            for i, c in enumerate(cols):
                tmp_s = re.findall('[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', c)
                if not tmp_s:
                    continue
                content.append(conv_num(tmp_s[0]))
            c_series = pd.Series(content, index = df.columns)
            df = df.append(c_series, ignore_index=True)
    return df

def read_json_log(json_log, column_list=['recon_loss', 'kld', 'loss']):
    '''reading .json file. Returns pandas dataframe.'''
    df = pd.DataFrame(columns=column_list)
    with open(json_log) as f:
        for single_data in json.load(f):
            content = list()
            for idx, loss_item in enumerate(single_data.items()):
                if loss_item[1] >= 1e50:
                    content.append(5000)
                    continue
                content.append(loss_item[1])
            c_series = pd.Series(content, index = df.columns)
            df = df.append(c_series, ignore_index=True)
    return df

def add_column_pd(df, c_name='loss'):
    '''adding a column to pandas dataframe.'''
    tmp = df[c_name]
    smoothed = smooth(df[c_name].tolist(), 0.95)
    df['smoothed_loss'] = smoothed
    return df

def conv_num(s: str):
    '''helper for converting str to number.'''
    try:
        return int(s)
    except ValueError:
        return float(s)

def smooth(scalars: List[float], weight: float) -> List[float]:
    '''exponential moving average for smoothing the axis.'''
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

### additional summary
def prep_dataframe(path: str):
    '''overall function.'''
    res_df = read_log(path)
    res_df = add_column_pd(res_df)
    return res_df

# testing section
parent = os.path.abspath(os.getcwd())
sys.path.append(parent)

# human_log = parent + '/models/ptc/human/log2021-08-22_12-09-31/log.txt'
# rat_log = parent + '/models/ptc/rat/log2021-08-22_12-05-06/log.txt'
# mouse_log = parent + '/models/ptc/mouseA/log2021-08-22_11-49-49/log.txt'
#
# human_df = prep_dataframe(human_log)
# rat_df = prep_dataframe(rat_log)
# mouse_df = prep_dataframe(mouse_log)

# fig, ax = plt.subplots(1, 3, figsize=(500, 5))
# # human section
# ax[0] = human_df.plot(x='iteration', y='smoothed_loss', ax=ax[0], legend=False, color='cornflowerblue')
# ax[0].set_title('Human dataset')
# ax[0].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
# ax0_xticks = ax[0].get_xticks()
# ax[0].set_xticks([ax0_xticks[1], ax0_xticks[-2]])
# ax0_yticks = ax[0].get_yticks()
# ax[0].set_yticks([ax0_yticks[1], ax0_yticks[-2]])
# #human_df.plot(x='iteration', y='loss', ax=ax[0].twinx(), legend=False, color='black')
#
# # mouse section
# ax[1] = mouse_df.plot(x='iteration', y='smoothed_loss', ax=ax[1], legend=False, color='cornflowerblue')
# ax[1].set_title('Mouse dataset')
# ax[1].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
# ax1_xticks = ax[1].get_xticks()
# ax[1].set_xticks([ax1_xticks[1], ax1_xticks[-2]])
# ax1_yticks = ax[1].get_yticks()
# ax[1].set_yticks([ax1_yticks[1], ax1_yticks[-2]])
# #mouse_df.plot(x='iteration', y='loss', ax=ax[1].twinx(), legend=False, color='r')
#
# # rat section
# ax[2] = rat_df.plot(x='iteration', y='smoothed_loss', ax=ax[2], legend=False, color='cornflowerblue')
# ax[2].set_title('Rat dataset')
# ax[2].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
# ax2_xticks = ax[2].get_xticks()
# ax[2].set_xticks([ax2_xticks[1], ax2_xticks[-2]])
# ax2_yticks = ax[2].get_yticks()
# ax[2].set_yticks([ax2_yticks[1], ax2_yticks[-2]])
# #rat_df.plot(x='iteration', y='loss', ax=ax[2].twinx(), legend=False, color='r')
#
# plt.show()

# sb.relplot(data=human_df, x='iteration', y='loss', kind='line')
# plt.show()

# human_log = parent + '/models/ptc/transfer/run_2021-09-15/log_2021-09-15_14-41-47_train/log.txt'
# human_df = read_log(human_log)
# print(human_df)
# fig, ax = plt.subplots(1,1)
# ax = human_df.plot(x='iteration', y='loss', ax=ax, legend=False, color='cornflowerblue')
# ax.set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
# ax_xticks = ax.get_xticks()
# ax.set_xticks([0, 50000, 100000, 150000])
# ax_yticks = ax.get_yticks()
# ax.set_yticks([ax_yticks[1], ax_yticks[-2]])
# #ax.set_yscale('log')
# plt.show()

###############################################################################
### VAE section
###############################################################################


vae_mouse_log_path = parent + '/models/vae/mouseA/run_2021-09-23/log.json'
vae_mouse_log_df = read_json_log(vae_mouse_log_path)
res_df = add_column_pd(vae_mouse_log_df, c_name='recon_loss')
#print(vae_mouse_log_df)
print(res_df)

fig, ax = plt.subplots(1, 2, figsize=(500, 1))
# human section
ax[0] = res_df.plot(y='smoothed_loss', ax=ax[0], legend=False, color='cornflowerblue')
ax[0].set_title('Reconstruction loss')
ax[0].set(xlabel='iteration step')
ax0_xticks = ax[0].get_xticks()
ax[0].set_xticks([ax0_xticks[1], ax0_xticks[-2]])
ax0_yticks = ax[0].get_yticks()
ax[0].set_yticks([ax0_yticks[1], ax0_yticks[-2]])
#human_df.plot(x='iteration', y='loss', ax=ax[0].twinx(), legend=False, color='black')

# mouse section
ax[1] = res_df.plot(y='kld', ax=ax[1], legend=False, color='cornflowerblue')
ax[1].set_title('Kulback-Leibler Divergence')
ax[1].set(xlabel='iteration step')
ax1_xticks = ax[1].get_xticks()
ax[1].set_xticks([ax1_xticks[1], ax1_xticks[-2]])
ax1_yticks = ax[1].get_yticks()
ax[1].set_yticks([ax1_yticks[1], ax1_yticks[-2]])
ax[1].set_yscale('log')
#mouse_df.plot(x='iteration', y='loss', ax=ax[1].twinx(), legend=False, color='r')

# rat section
# ax[2] = res_df.plot(y='smoothed_loss', ax=ax[2], legend=False, color='cornflowerblue')
# ax[2].set_title('Total Loss')
# ax[2].set(xlabel='iteration step')
# ax2_xticks = ax[2].get_xticks()
# ax[2].set_xticks([ax2_xticks[1], ax2_xticks[-2]])
# ax2_yticks = ax[2].get_yticks()
# ax[2].set_yticks([ax2_yticks[1], ax2_yticks[-2]])
# ax[2].set_yscale('log')
# rat_df.plot(x='iteration', y='loss', ax=ax[2].twinx(), legend=False, color='r')

plt.show()
