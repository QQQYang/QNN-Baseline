"""
Visualiza the experiment results about effect of batch size on the training of QNN (Figure 13 in the paper)
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
    'GD_4_0_0.npy',
    'GD_8_0_0.npy',
    'GD_16_0_0.npy',
    'GD_32_0_0.npy',
    'GD_64_0_0.npy',
]

legend = [
    'bs=4',
    'bs=8',
    'bs=16',
    'bs=32',
    'bs=64',
]

# colors = {
#     'train': ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0'],
#     'test': ['#7e1e9c80', '#15b01a80', '#0343df80', '#ff81c080']
# }

colors = {
    'train': ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728'],
    'test': ['#1f78b480', '#ff7f0e80', '#2ca02c80', '#9467bd80', '#8c564b80', '#e377c280', '#7f7f7f80', '#bcbd2280', '#17becf80', '#d6272880']
}

markers = ["o", "s", "^", '+', 'd']
letter = list(map(chr, range(ord('a'), ord('z') + 1)))
title = ['QNNN', 'QENN']

bbox_to_anchor = {
    'qnn': (0.4, 0.1, 1, 1),
    'qnn_embedding': (0.5, 0.1, 1, 1)
}
xlim = {
    'qnn': [30, 40],
    'qnn_embedding': [30, 40],
}
ylim = {
    'qnn': [0.75, 1.0],
    'qnn_embedding': [0.85, 1.0]
}
window_size = {
    'qnn': ["50%", "30%"],
    'qnn_embedding': ["40%", "40%"],
}

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

root = 'logs'

yticks = [0, 0.4, 0.6, 0.8, 1.0]
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10.08, 4.8))
lines, lines_label = [], []
for k, net in enumerate(['qnn', 'qnn_embedding']):
    # plt.yticks(range(len(yticks)), yticks)
    # set sub figure
    axins = inset_axes(ax[k], width=window_size[net][0], height=window_size[net][1], loc='lower left', bbox_to_anchor=bbox_to_anchor[net], bbox_transform=ax[k].transAxes)
    complete_flag = False
    for j, config in enumerate(results):
        for mode in ['train', 'test']:
            data = []
            for i in range(1):
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
            line = ax[k].plot(result[::2], color=colors[mode][j], marker=markers[j], markersize=3, linewidth=0.5)[0]
            ax[k].tick_params(labelsize=20)
            lines.append(line)
            lines_label.append(mode+'('+legend[j]+')')

            axins.plot(result[::2], color=colors[mode][j], marker=markers[j], markersize=3, linewidth=0.5)[0]
            complete_flag = True
    # ax[k].set_title('('+letter[k]+') '+title[k], y=-0.12, fontsize=15)
    ax[k].set_title(title[k], fontsize=20)
    ax[k].set_xlabel('Epoch', fontsize=20)
    if k == 0:
        ax[k].set_ylabel('Accuracy', fontsize=20)
    axins.set_xlim(xlim[net][0], xlim[net][1])
    axins.set_ylim(ylim[net][0], ylim[net][1])
    axins.tick_params(labelsize=15)
    mark_inset(ax[k], axins, loc1a=1, loc1b=4, loc2a=2, loc2b=3, fc="none", ec='k', lw=1)
plt.tight_layout()
fig.legend(lines[:len(lines)//2], labels=lines_label[:len(lines_label)//2], loc='center right', borderaxespad=0.2, fontsize=17)#, labelspacing=0.5)
plt.subplots_adjust(right=0.79)
plt.savefig('test/net_bSGD.pdf', dpi=600, format='pdf')
plt.show()
plt.clf()

# draw curve of accuracy vs batch size
xticks = [4, 8, 16, 32, 64]
plt.xscale('log', base=2)
for i, net in enumerate(['qnn', 'qnn_embedding']):
    data = []
    for j, config in enumerate(results):
        result_path = os.path.join(root, net, str(0), '_'.join(['acc', 'test', net, config]))
        result = np.load(result_path)
        data.append(result[-1])
    plt.plot(xticks, data, label=title[i], color=colors['test'][i], marker=markers[i], markersize=4)
plt.tick_params(labelsize=20)
plt.legend(loc='best', fontsize=20)
plt.xlabel('Batch size', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.tight_layout()
plt.savefig('figure/acc_bs.pdf', dpi=600, format='pdf')
plt.show()
plt.clf()