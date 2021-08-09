"""
Visualiza the experiment results about the effect of noise in NISQ era (Figure 11 in the paper)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import torch

results = [
    'GD_4_0_0.npy',
    'GD_4_0_0_noisy.npy',
    'GD_4_0_0-random_pure.npy',
    'GD_4_0_0-random_noisy.npy',
]

legend = [
    'FT, TL',
    'NISQ, TL',
    'FT, RL',
    'NISQ, RL',
]

colors = {
    'train': ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728'],
    'test': ['#1f78b480', '#ff7f0e80', '#2ca02c80', '#9467bd80', '#8c564b80', '#e377c280', '#7f7f7f80', '#bcbd2280', '#17becf80', '#d6272880']
}

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

markers = ["o", "s", "^", '+', 'd']
letter = list(map(chr, range(ord('a'), ord('z') + 1)))
title = ['QNNN', 'QENN']
linestyle = ['-', '--']

root = 'logs'

yticks = [0, 0.4, 0.6, 0.8, 1.0]
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10.08, 4.0))
x = range(0, 100, 10)
lines, lines_label = [], []
for k, net in enumerate(['qnn', 'qnn_embedding']):
    # plt.yticks(range(len(yticks)), yticks)
    # set sub figure
    # axins = inset_axes(ax[k], width=window_size[net][0], height=window_size[net][1], loc='lower left', bbox_to_anchor=bbox_to_anchor[net], bbox_transform=ax[k].transAxes)
    complete_flag = False
    for j, config in enumerate(results):
        for mode in ['train', 'test']:
            data = []
            for i in range(10):
                result_path = os.path.join(root, net, str(i), '_'.join(['acc', mode, net, config]))
                if not os.path.exists(result_path):
                    complete_flag = False
                    continue
                result = np.load(result_path)
                if not isinstance(result.tolist(), list) or len(result) < 2:
                    complete_flag = False
                    continue
                if len(result) < 100:
                    result = result.tolist()
                    result = result + [result[-1]]*(100 - len(result))
                    result = np.array(result)
                data.append(result)
            result = np.mean(data, axis=0)
            # new_y = interpolate.interp1d(yticks, range(len(yticks)))(result[::2])
            line = ax[k].plot(x, result[::10], color=colors[mode][j//2], marker=markers[j], linestyle=linestyle[j%2])[0]
            ax[k].tick_params(labelsize=15)
            lines.append(line)
            lines_label.append(mode+'('+legend[j]+')')

            # axins.plot(result[::2], color=colors[mode][j], marker=markers[j], markersize=3, linewidth=0.5)[0]
            complete_flag = True
    # ax[k].set_title('('+letter[k]+') '+title[k], y=-0.12, fontsize=15)
    ax[k].set_title(title[k], fontsize=20)
    ax[k].set_xlabel('Epoch', fontsize=20)
    ax[k].grid(True)
    if k == 0:
        ax[k].set_ylabel('Accuracy', fontsize=20)
    # axins.set_xlim(xlim[net][0], xlim[net][1])
    # axins.set_ylim(ylim[net][0], ylim[net][1])
    # mark_inset(ax[k], axins, loc1a=1, loc1b=4, loc2a=2, loc2b=3, fc="none", ec='k', lw=1)
plt.tight_layout()
fig.legend(lines[:len(lines)//2], labels=lines_label[:len(lines_label)//2], loc='center right', borderaxespad=0.2, fontsize=15, labelspacing=0.5)
plt.subplots_adjust(right=0.78)
plt.savefig('figure/NISQ.pdf', dpi=600, format='pdf')
plt.show()
plt.clf()