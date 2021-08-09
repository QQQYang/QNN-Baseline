#encoding=utf-8
"""
Evaluation of benchmark
--------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from prettytable import PrettyTable

def vis(root, name):
    root = os.path.join(root, name)
    instances = os.listdir(root)
    results = {'acc': {}, 'loss': {}}
    for instance in instances:
        instance_path = os.path.join(root, instance)
        configs = os.listdir(instance_path)
        for config in configs:
            if 'acc_test' in config and 'wine' in config:
                config_path = os.path.join(instance_path, config)
                data = np.load(config_path).tolist()
                prefix = config.split('_')[0]
                suffix = '_'.join(config.split('_')[1:])
                if suffix not in results[prefix]:
                    results[prefix][suffix] = [data]
                else:
                    results[prefix][suffix].append(data)

    #fig, ax = plt.subplots(figsize=(110, 100))
    colors=['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#95d0fc', '#029386']
    cnt = 0
    #for key in results['loss']:
    #    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    #    data = np.array(results['loss'][key])
    #    mean = np.mean(data, axis=0)
    #    std = np.std(data, axis=0)
    #    plt.plot(range(len(mean)), mean, color=colors[cnt%len(colors)], label=os.path.splitext(key)[0])
    #    #plt.fill_between(range(len(mean)), mean+std, mean-std, facecolor=colors[cnt%len(colors)], alpha=0.5)
    #    cnt += 1
    #plt.legend(loc='best')
    #plt.title('loss vs batch-size')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.savefig(os.path.join('logs', name+'_loss.png'))
    #plt.clf()

    table = PrettyTable(['config', 'acc'])
    for key in results['acc']:
        acc = round(np.mean(np.array(results['acc'][key])), 3)
        table.add_row([os.path.splitext(key)[0], str(acc)])
    print(table)

if __name__ == '__main__':
    root = 'logs'
    nets = ['qnn', 'qnn_embedding']
    for net in nets:
        vis(root, net)
