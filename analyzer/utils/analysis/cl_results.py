import os, sys
import matplotlib.pyplot as plt
import pandas as pd
parent = os.path.abspath(os.getcwd())
sys.path.append(parent)
from analyzer.utils.analysis.model_states import read_log, read_log_over_double_lines

def read_log_cl_results(txt_file_train, txt_file_test):
    '''adding acc from log/txt file to pandas dataframe.'''
    df = read_log_over_double_lines(txt_file_train, column_list=['iteration', 'loss', 'lr', 'epoch'])
    testdf = read_log(txt_file_test, column_list=['accuracy'])

    #df['accuracy'] = testdf
    return df, testdf

train_log = parent + '/models/cl/mouseA/run_2021-10-11/log_2021-10-11_09-35-07_train/log.txt'
test_log = parent + '/models/cl/mouseA/run_2021-10-11/log_2021-10-11_09-35-07_test/log.txt'

df, testdf = read_log_cl_results(train_log, test_log)

fig, ax = plt.subplots(1, 3, figsize=(500, 1))
# human section
ax[0] = df.plot(y='loss', ax=ax[0], legend=False, color='cornflowerblue')
ax[0].set_title('Training loss')
ax[0].set(xlabel='iteration step')
ax0_xticks = ax[0].get_xticks()
ax[0].set_xticks([ax0_xticks[1], ax0_xticks[-2]])
ax0_yticks = ax[0].get_yticks()
ax[0].set_yticks([ax0_yticks[1], ax0_yticks[-2]])

ax[1] = df.plot(y='lr', ax=ax[1], legend=False, color='cornflowerblue')
ax[1].set_title('Learning rate')
ax[1].set(xlabel='iteration step')
ax1_xticks = ax[1].get_xticks()
ax[1].set_xticks([ax1_xticks[1], ax1_xticks[-2]])
ax1_yticks = ax[1].get_yticks()
ax[1].set_yticks([ax1_yticks[1], ax1_yticks[-2]])

ax[2] = testdf.plot(y='accuracy', ax=ax[2], legend=False, color='cornflowerblue')
ax[2].set_title('accuracy')
ax[2].set(xlabel='epoch')
ax2_xticks = ax[2].get_xticks()
ax[2].set_xticks([ax2_xticks[1], ax2_xticks[-2]])
ax2_yticks = ax[2].get_yticks()
ax[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

plt.show()
