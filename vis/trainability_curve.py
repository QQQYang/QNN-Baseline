"""
Visualiza the experiment results about QNN's generalization (Figure 3 and Figure 4 in the paper)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, TransformedBbox, BboxPatch, BboxConnector
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

data = {
    'synthetic':{
        'true':[
            'qnn/0/qnn_GD_4_0_0.npy',
            'qnn_embedding/0/qnn_embedding_GD_4_0_0.npy',
            'mlp/0/mlp_true-synthetic.npy'
        ],
        'random': [
            'qnn/0/qnn_GD_4_0_0-random_pure.npy',
            'qnn_embedding/0/qnn_embedding_GD_4_0_0-random_pure.npy',
            'mlp/0/mlp_random-synthetic.npy',
            'mlp/0/mlp_random-synthetic-plus.npy'
        ]
    },
    'wine':{
        'true': [
            'qnn/0/qnn_GD_4_0_0-wine.npy',
            'qnn_embedding/0/qnn_embedding_GD_4_0_0-wine.npy',
            'mlp/0/mlp_true_wine.npy'
        ],
        'random': [
            'qnn/0/qnn_GD_4_0_0-random_pure-wine.npy',
            'qnn_embedding/0/qnn_embedding_GD_4_0_0-random_pure-wine.npy',
            'mlp/0/mlp_random_wine.npy'
        ]
    },
    'MNIST':{
        'true':[
            'qcnn/0/MLP_true.npy',
            'qcnn/0/QCNN_true.npy',
            'qcnn/0/CNN_true.npy'
        ],
        'random':[
            'qcnn/0/MLP_random.npy',
            'qcnn/0/QCNN_random.npy',
            'qcnn/0/CNN_random.npy',
            'qcnn/0/MLP_random_plus.npy'
        ]
    }
}

legend = {
    'synthetic':[
        'QNNN',
        'QENN',
        'MLP',
        'MLP++',
    ],
    'wine':[
        'QNNN',
        'QENN',
        'MLP',
        'MLP++',
    ],
    'MNIST':[
        'MLP',
        'QCNN',
        'CNN',
        'MLP++',
    ],
}

legend2index = {
    'QNNN': 0,
    'QENN': 1,
    'MLP': 2,
    'MLP++': 3,
    'QCNN': 4,
    'CNN': 5,
}

root = 'logs'

# colors = {
#     'train': ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0'],
#     'test': ['#7e1e9c80', '#15b01a80', '#0343df80', '#ff81c080']
# }
colors = {
    'train': ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728'],
    'test': ['#1f78b480', '#ff7f0e80', '#2ca02c80', '#9467bd80', '#8c564b80', '#e377c280', '#7f7f7f80', '#bcbd2280', '#17becf80', '#d6272880']
}
shape = ["o", "s", "^", '+', 'd', '|']
letter = list(map(chr, range(ord('a'), ord('z') + 1)))

window_size = {
    'true': {
        'synthetic': ["60%", "40%"],
        'wine': ["60%", "35%"],
        'MNIST': ["50%", "50%"],
    },
    'random': {
        'MNIST': ["50%", "50%"]
    }
}

bbox_to_anchor = {
    'true': {
        'synthetic': (0.3, 0.3, 1, 1),
        'wine': (0.3, 0.12, 1, 1),
        'MNIST': (0.4, 0.1, 1, 1)
    },
    'random': {
        'MNIST': (0.4, 0.2, 1, 1)
    }
}

xlim = [30, 40]
ylim = [0.09, 0.14]

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

title = ['Quantum', 'Wine', 'MNIST']

x = range(0, 100, 5)
for label in ['true', 'random']:
    fig, ax = plt.subplots(1, 3, sharex='col', figsize=(12.8, 5.5))
    line_labels = {}
    for k, dataset in enumerate(['synthetic', 'wine', 'MNIST']):
        if label == 'true' or dataset == 'MNIST':
            axins = inset_axes(ax[k], width=window_size[label][dataset][0], height=window_size[label][dataset][1], loc='lower left', bbox_to_anchor=bbox_to_anchor[label][dataset], bbox_transform=ax[k].transAxes)
        cnt = 0
        gen_err = []
        for i, model in enumerate(data[dataset][label]):
            prefix = model.split('/')[0]
            suffix = model.split('/')[-1]
            train, test = 0, 0
            for phase in ['train', 'test']:
                result = np.load(os.path.join(root, prefix, str(0), '_'.join(['acc', phase, suffix])))
                if dataset != 'MNIST':
                    result = result[::5]
                    x = range(0, 100, 5)
                else:
                    x = range(0, len(result), 2)
                    if i < 3:
                        result = result[::2]
                    else:
                        result = result[:200][::8]
                        x = range(0, len(result)*2, 2)
                line, = ax[k].plot(x, result, label=legend[dataset][i]+'('+phase+')', color=colors[phase][legend2index[legend[dataset][i]]], marker=shape[legend2index[legend[dataset][i]]])
                if legend[dataset][i]+'('+phase+')' not in line_labels:
                    line_labels[legend[dataset][i]+'('+phase+')'] = line
                ax[k].tick_params(labelsize=20)
                if phase == 'train':
                    train = result[-1]
                else:
                    test = result[-1]
                
                if dataset == 'MNIST' and label == 'random':
                    axins.plot(x, result, color=colors[phase][legend2index[legend[dataset][i]]], marker=shape[legend2index[legend[dataset][i]]])
                    axins.tick_params(labelsize=15)
            cnt += 1
            gen_err.append(abs(train - test))
        if label == 'true':
            axins.bar(range(len(gen_err)), gen_err, color='#929591')
            axins.set_xticks(range(len(gen_err)))
            axins.set_xticklabels(legend[dataset][:len(gen_err)])
            axins.set_ylabel('G-Error', fontsize=15)
            axins.tick_params(labelsize=15)
        elif dataset == 'MNIST':
            axins.set_xlim(xlim[0], xlim[1])
            axins.set_ylim(ylim[0], ylim[1])
            mark_inset(ax[k], axins, loc1a=3, loc1b=2, loc2a=4, loc2b=1, fc="none", ec='k', lw=1, linestyle='--')
        # ax[k].legend(loc='best')
        ax[k].set_title('('+letter[k]+') '+title[k], fontsize=20)
        ax[k].grid(True)
        ax[k].set_xlabel('Epoch', fontsize=20)
        if k == 0:
            ax[k].set_ylabel('Accuracy', fontsize=20)
    plt.tight_layout()
    fig.legend(handles=list(line_labels.values()), labels=list(line_labels.keys()), loc='upper left', ncol=4, fontsize=20, borderaxespad=0.2, mode="expand")
    plt.subplots_adjust(top=0.68)
    plt.savefig('figure/sec3_'+label+'.pdf', dpi=600, format='pdf')
    plt.show()
        # plt.cla()