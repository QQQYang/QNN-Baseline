"""
Visualiza the experiment results about QNN's trainability (Figure 5 (b) in the paper)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

data = {
    'train': [],
    'test': []
}
for i in range(10):
    result_path_test = os.path.join('logs/qnn_embedding', str(i), 'acc_test_qnn_embedding_GD_200_0_0-wine_e500.npy')
    result_path_train = os.path.join('logs/qnn_embedding', str(i), 'acc_train_qnn_embedding_GD_200_0_0-wine_e500.npy')
    test = np.load(result_path_test)
    train = np.load(result_path_train)
    data['train'].append(train[::20])
    data['test'].append(test[::20])

colors = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']

markers = ["o", "s", "^", '+', 'd']

legend = [
    'GD',
    'SGD',
    'SGD+WD',
    'SGD+ES',
]

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

plt.figure(figsize=(6.4, 5.4))
mean_final = []
exp_min = 1
for i, phase in enumerate(['train', 'test']):
    mean = np.mean(data[phase], axis=0)
    std = np.std(data[phase], axis=0)
    x = range(0, 500, 20)
    plt.plot(x, mean, label=phase+'('+legend[0]+')', color=colors[i], marker=markers[i])
    plt.plot(x, [mean[-1]]*len(x), color=colors[i], linestyle='--')
    mean_final.append(mean[-1])
    exp_min = min(np.min(mean-std), exp_min)
    plt.fill_between(x, mean+std, mean-std, color=colors[i], alpha=0.1)
    plt.plot(x, mean+std, color=colors[i], linestyle='--', alpha=0.5)
    plt.plot(x, mean-std, color=colors[i], linestyle='--', alpha=0.5)

data = {
    'train': [],
    'test': []
}
for i in range(10):
    result_path_test = os.path.join('logs/qnn_embedding', str(i), 'acc_test_qnn_embedding_GD_4_0_0-wine.npy')
    result_path_train = os.path.join('logs/qnn_embedding', str(i), 'acc_train_qnn_embedding_GD_4_0_0-wine.npy')
    test = np.load(result_path_test)
    train = np.load(result_path_train)
    data['train'].append(train)
    data['test'].append(test)

x = range(0, 100, 10)
for i, phase in enumerate(['train', 'test']):
    mean = np.mean(data[phase], axis=0)
    plt.plot(x, mean[::10], label=phase+'('+legend[1]+')', color=colors[i+2], marker=markers[i+2])
    # y = np.arange(exp_min, mean_final[i], 0.001)
    # plt.plot([x[np.where(mean-mean_final[i]<0.001)[0][-1]]]*len(y), y, color=colors[i+2], linestyle='--', alpha=0.5)

# plt.legend(loc='best', fontsize=15)
plt.legend(loc='lower left', ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), fontsize=20, borderaxespad=0.0, mode="expand")
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('figure/gd_e500.pdf', dpi=600, format='pdf')
plt.show()