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
            'qnn/0/qnn_GD_4_0_0-wine-qas.npy',
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
        'QNNN_QAS',
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
    'QNNN_QAS': 6
}

root = 'logs'

# colors = {
#     'train': ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0'],
#     'test': ['#7e1e9c80', '#15b01a80', '#0343df80', '#ff81c080']
# }
colors = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (44, 160, 44), (163, 31, 52), (148, 103, 189), (23, 190, 207)]
alpha = {
    'train': 1.0,
    'test': 0.5
}
linestyle = {
    'train': '-',
    'test': '--'
}
shape = ["o", "s", "^", '+', 'd', '|', '>']
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

# plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

title = ['Quantum', 'Wine', 'MNIST']

x = range(0, 100, 5)
for label in ['true']:
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 8))
    gen_errs = []
    for k, dataset in enumerate(['synthetic', 'wine', 'MNIST']):
        line_labels = {}
        if label == 'random' and dataset == 'MNIST':
            axins = inset_axes(ax[k//2][k%2], width=window_size[label][dataset][0], height=window_size[label][dataset][1], loc='lower left', bbox_to_anchor=bbox_to_anchor[label][dataset], bbox_transform=ax[k//2][k%2].transAxes)
        cnt = 0
        gen_err = []
        for i, model in enumerate(data[dataset][label]):
            prefix = model.split('/')[0]
            suffix = model.split('/')[-1]
            train, test = 0, 0
            for phase in ['train', 'test']:
                result = np.load(os.path.join(root, prefix, str(0), '_'.join(['acc', phase, suffix])))
                if dataset != 'MNIST':
                    result = result[::10]
                    x = range(0, 100, 10)
                else:
                    x = range(0, len(result), 5)
                    if i < 3:
                        result = result[::5]
                    else:
                        result = result[:200][::20]
                        x = range(0, len(result)*5, 5)
                    # x = np.array(x)*2
                line, = ax[k//2][k%2].plot(x, result, label=legend[dataset][i]+'('+phase+')', color=np.array(colors[legend2index[legend[dataset][i]]])/255, marker=shape[legend2index[legend[dataset][i]]], alpha=alpha[phase], linestyle=linestyle[phase])
                if legend[dataset][i]+'('+phase+')' not in line_labels:
                    line_labels[legend[dataset][i]+'('+phase+')'] = line
                ax[k//2][k%2].tick_params(labelsize=20)
                if phase == 'train':
                    train = result[-1]
                else:
                    test = result[-1]
                
                if dataset == 'MNIST' and label == 'random':
                    axins.plot(x, result, color=np.array(colors[legend2index[legend[dataset][i]]])/255, marker=shape[legend2index[legend[dataset][i]]], alpha=alpha[phase], linestyle=linestyle[phase])
                    axins.tick_params(labelsize=15)
            cnt += 1
            gen_err.append(abs(train - test))
        if dataset == 'MNIST' and label == 'random':
            axins.set_xlim(xlim[0], xlim[1])
            axins.set_ylim(ylim[0], ylim[1])
            mark_inset(ax[k//2][k%2], axins, loc1a=3, loc1b=2, loc2a=4, loc2b=1, fc="none", ec='k', lw=1, linestyle='--')
        gen_errs.append(gen_err)
        # ax[k].legend(loc='best')
        ax[k//2][k%2].set_title('('+letter[k]+') '+title[k], fontsize=20)
        ax[k//2][k%2].grid(True)
        ax[k//2][k%2].set_xlabel('Epoch', fontsize=20)
        if k%2 == 0:
            ax[k//2][k%2].set_ylabel('Accuracy', fontsize=20)
        
        value = list(line_labels.values())
        value = value[::2]+value[1:len(value):2]
        key = list(line_labels.keys())
        key = key[::2]+key[1:len(key):2]
        ax[k//2][k%2].legend(handles=value, labels=key, loc='best', ncol=2, fontsize=15)

    # draw the generalization error
    bar_labels = {}
    for k, dataset in enumerate(['synthetic', 'wine', 'MNIST']):
        err = gen_errs[k]
        for j in range(len(err)):
            bar = ax[1][1].bar(j+k*len(err)+k, err[j], edgecolor='k', fc=np.array(colors[legend2index[legend[dataset][j]]])/255)
            ax[1][1].text(j+k*len(err)+k-0.45, err[j]+0.008, str(round(err[j], 3)), fontsize=15)
            if legend[dataset][j] not in bar_labels:
                bar_labels[legend[dataset][j]] = bar
    ax[1][1].set_xticks([1, 5, 9])
    ax[1][1].set_xticklabels(title)
    ax[1][1].legend(handles=list(bar_labels.values()), labels=list(bar_labels.keys()), loc='best', ncol=2, fontsize=15)
    ax[1][1].tick_params(labelsize=20)
    ax[1][1].set_ylim([0.0, 0.45])
    ax[1][1].set_ylabel('G-Error', fontsize=15)

    plt.tight_layout()
    # fig.legend(handles=list(line_labels.values()), labels=list(line_labels.keys()), loc='upper left', ncol=4, fontsize=20, borderaxespad=0.2, mode="expand")
    # plt.subplots_adjust(top=0.68)
    # plt.savefig('figure/sec3_'+label+'_new.pdf', dpi=600, format='pdf')
    plt.show()
        # plt.cla()