import numpy as np
import os

data_file = {
    'wine': {
        'true': [
            'qnn_GD_4_0_0-wine.npy',
            'qnn_GD_4_0_0-wine-normal.npy',
            'qnn_embedding_GD_4_0_0-wine.npy',
            'qnn_embedding_GD_4_0_0-wine-normal.npy',
        ],
        'random': [
            'qnn_GD_4_0_0-random_pure-wine.npy',
            'qnn_GD_4_0_0-random_pure-wine-normal.npy',
            'qnn_embedding_GD_4_0_0-random_pure-wine.npy',
            'qnn_embedding_GD_4_0_0-random_pure-wine-normal.npy',
        ]
    }
}

for label in ['true', 'random']:
    for dataset in ['wine']:
        for mode in data_file[dataset][label]:
            prefix = mode.split('_')
            if 'embedding' in mode:
                prefix = '_'.join(prefix[:2])
            else:
                prefix = prefix[0]
            data = {'train': [], 'test': []}
            for seed in range(10):
                for phase in ['train', 'test']:
                    data[phase].append(np.load(os.path.join('logs', prefix, str(seed), '_'.join(['acc', phase, mode]))))
            acc_train = np.mean(data['train'], axis=0)
            acc_test = np.mean(data['test'], axis=0)
            print('Mode = {}, train = {}, test = {}'.format(mode, acc_train[-1], acc_test[-1]))