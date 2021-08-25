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

def read_log(txt_file):
    '''reading log/txt file. Returns pandas dataframe.'''
    df = pd.DataFrame(columns=['iteration', 'loss', 'lr'])
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

def add_column_pd(df):
    '''adding a column to pandas dataframe.'''
    tmp = df['loss']
    smoothed = smooth(df['loss'].tolist(), 0.95)
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

human_log = parent + '/models/ptc/human/log2021-08-22_12-09-31/log.txt'
rat_log = parent + '/models/ptc/rat/log2021-08-22_12-05-06/log.txt'
mouse_log = parent + '/models/ptc/mouseA/log2021-08-22_11-49-49/log.txt'

human_df = prep_dataframe(human_log)
rat_df = prep_dataframe(rat_log)
mouse_df = prep_dataframe(mouse_log)

fig, ax = plt.subplots(1, 3, figsize=(500, 5))
# human section
ax[0] = human_df.plot(x='iteration', y='smoothed_loss', ax=ax[0], legend=False, color='cornflowerblue')
ax[0].set_title('Human dataset')
ax[0].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
ax0_xticks = ax[0].get_xticks()
ax[0].set_xticks([ax0_xticks[1], ax0_xticks[-2]])
ax0_yticks = ax[0].get_yticks()
ax[0].set_yticks([ax0_yticks[1], ax0_yticks[-2]])
#human_df.plot(x='iteration', y='loss', ax=ax[0].twinx(), legend=False, color='black')

# mouse section
ax[1] = mouse_df.plot(x='iteration', y='smoothed_loss', ax=ax[1], legend=False, color='cornflowerblue')
ax[1].set_title('Mouse dataset')
ax[1].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
ax1_xticks = ax[1].get_xticks()
ax[1].set_xticks([ax1_xticks[1], ax1_xticks[-2]])
ax1_yticks = ax[1].get_yticks()
ax[1].set_yticks([ax1_yticks[1], ax1_yticks[-2]])
#mouse_df.plot(x='iteration', y='loss', ax=ax[1].twinx(), legend=False, color='r')

# rat section
ax[2] = rat_df.plot(x='iteration', y='smoothed_loss', ax=ax[2], legend=False, color='cornflowerblue')
ax[2].set_title('Rat dataset')
ax[2].set(xlabel='iteration step', ylabel='training loss [ChamferDistance]')
ax2_xticks = ax[2].get_xticks()
ax[2].set_xticks([ax2_xticks[1], ax2_xticks[-2]])
ax2_yticks = ax[2].get_yticks()
ax[2].set_yticks([ax2_yticks[1], ax2_yticks[-2]])
#rat_df.plot(x='iteration', y='loss', ax=ax[2].twinx(), legend=False, color='r')

plt.show()

# sb.relplot(data=human_df, x='iteration', y='loss', kind='line')
# plt.show()
