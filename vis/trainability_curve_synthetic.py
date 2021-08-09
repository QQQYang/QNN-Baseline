"""
Visualiza the experiment results about QNN's trainability (Figure 12 in the paper)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, TransformedBbox, BboxPatch, BboxConnector
from scipy import interpolate
import os

def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

results = [
    'GD_200_0_0.npy',
    'GD_4_0_0.npy',
    'GD_4_5e-5_0.npy',
    # 'GD_4_0_10.npy',
    'GD_4_0_0_noisy.npy'
]

legend = [
    'GD',
    'SGD',
    'SGD+WD',
    # 'SGD+ES',
    'SGD+N'
]

# colors = {
#     'train': ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0'],
#     'test': ['#7e1e9c80', '#15b01a80', '#0343df80', '#ff81c080']
# }

colors = {
    'train': ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728'],
    'test': ['#1f78b480', '#ff7f0e80', '#2ca02c80', '#9467bd80', '#8c564b80', '#e377c280', '#7f7f7f80', '#bcbd2280', '#17becf80', '#d6272880']
}

markers = ["o", "s", "^", '+', 'd', '|']
letter = list(map(chr, range(ord('a'), ord('z') + 1)))
title = ['QNNN', 'QENN']

bbox_to_anchor = {
    'qnn': (0.2, 0.35, 1, 1),
    'qnn_embedding': (0.4, 0.2, 1, 1.0)
}
xlim = {
    'qnn': [20, 40],
    'qnn_embedding': [20, 40],
}
ylim = {
    'qnn': [0.92, 0.98],
    'qnn_embedding': [0.8, 0.92]
}
window_size = {
    'qnn': ["40%", "30%"],
    'qnn_embedding': ["40%", "40%"],
}

loc = {
    'qnn': [2, 3, 1, 4],
    'qnn_embedding': [1, 1, 3, 3]
}

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

root = 'logs'

# plt.rcParams['lines.dashed_pattern'] = [5.0, 5.0]

yticks = [0.2, 0.85, 0.90, 0.95, 1.0]
yticks_trans = interpolate.interp1d(yticks, range(len(yticks)))
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10.08, 4.0))
lines, lines_label = [], []
x = range(0, 101, 10)
for k, net in enumerate(['qnn', 'qnn_embedding']):
    # ax[k].set_yticks(range(len(yticks)))
    # ax[k].set_yticklabels(yticks)
    # if k == 1:
    axins = inset_axes(ax[k], width=window_size[net][0], height=window_size[net][1], loc='lower left', bbox_to_anchor=bbox_to_anchor[net], bbox_transform=ax[k].transAxes)
        # plt.setp(list(axins.spines.values()), linestyle="--")
    #     axins.set_yticks(range(len(yticks)))
    #     axins.set_yticklabels(yticks)
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
                if not isinstance(result.tolist(), list) or len(result) < 10:
                    complete_flag = False
                    continue
                if len(result) < 100 and 'noisy' not in config:
                    result = result.tolist()
                    result = result + [result[-1]]*(100 - len(result))
                    result = np.array(result)
                data.append(result)
            result = np.mean(data, axis=0)
            # new_y = yticks_trans(result[::10])
            if 'noisy' in config:
                line = ax[k].plot(range(0, 21, 4), result[::4].tolist()+[result[-1].tolist()], label=mode+'('+legend[j]+')', color=colors[mode][j], marker=markers[j])[0]
            else:
                line = ax[k].plot(x, result[::10].tolist()+[result[-1].tolist()], label=mode+'('+legend[j]+')', color=colors[mode][j], marker=markers[j])[0]
            ax[k].tick_params(labelsize=15)
            lines.append(line)
            lines_label.append(mode+'('+legend[j]+')')

            # if k == 1:
            if 'noisy' not in config:
                axins.plot(x, result[::10].tolist()+[result[-1].tolist()], color=colors[mode][j], marker=markers[j])
                axins.tick_params(labelsize=15)
            complete_flag = True
    # plt.legend(loc='best')
    ax[k].set_title(title[k], fontsize=20)
    ax[k].set_xlabel('Epoch', fontsize=20)
    ax[k].grid(True)
    if k == 0:
        ax[k].set_ylabel('Accuracy', fontsize=20)
    # if k == 1:
        # ylim[net] = yticks_trans(ylim[net])
    axins.set_xlim(xlim[net][0], xlim[net][1])
    axins.set_ylim(ylim[net][0], ylim[net][1])
    mark_inset(ax[k], axins, loc1a=loc[net][0], loc1b=loc[net][1], loc2a=loc[net][2], loc2b=loc[net][3], fc="none", ec='k', lw=1, linestyle='--')
plt.tight_layout()
fig.legend(lines[:len(lines)//2], labels=lines_label[:len(lines_label)//2], loc='center right', borderaxespad=0.2, fontsize=15, labelspacing=0.5)
plt.subplots_adjust(right=0.78)
plt.savefig('figure/net_generalization.pdf', dpi=600, format='pdf')
plt.show()
plt.clf()