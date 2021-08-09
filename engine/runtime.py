"""
Running time of models
-------------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.utils import _flatten, unflatten
from pennylane.optimize import NesterovMomentumOptimizer, GradientDescentOptimizer, AdamOptimizer, RMSPropOptimizer, QNGOptimizer

import qiskit
import qiskit.providers.aer.noise as noise

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

import yaml
from easydict import EasyDict
import argparse
import os
import sys
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, TransformedBbox, BboxPatch, BboxConnector
import json

sys.path.append('.')

from circuit import build_circuit, MLP
from tools import AverageMeter
from data_gen import data_gen, get_data
from tools.metric import cost, accuracy, build_optimizer, build_scheduler
from tools.logger import setup_logger

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

def get_config(config_file):
    """
    Read config from config yaml
    """
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    return config

class QNN:
    def __init__(self, cfg):
        net = build_circuit(cfg)
        self.opt = GradientDescentOptimizer(cfg.TRAIN.LR)
        if cfg.CIRCUIT.BACKEND == 'qiskit.aer':
            provider = qiskit.IBMQ.providers(group='open')[0]
            backend = provider.get_backend('ibmq_16_melbourne')    # ibmq_qasm_simulator
            noise_model = noise.NoiseModel.from_backend(backend)

            dev = qml.device(cfg.CIRCUIT.BACKEND, wires=cfg.CIRCUIT.N_QUBIT, noise_model=noise_model, shots=10)
        else:
            dev = qml.device(cfg.CIRCUIT.BACKEND, wires=cfg.CIRCUIT.N_QUBIT)
        self.circuit = qml.QNode(net, dev)

        param = np.random.uniform(0, math.pi, cfg.CIRCUIT.N_QUBIT*cfg.CIRCUIT.N_DEPTH*3)
        self.param = np.reshape(param, (cfg.CIRCUIT.N_DEPTH, cfg.CIRCUIT.N_QUBIT, 3))
        self.feat = np.random.uniform(0, 1, cfg.CIRCUIT.N_QUBIT*4)
        self.feat = np.reshape(self.feat, (4, 1, cfg.CIRCUIT.N_QUBIT))
        self.label = np.array([-1, 1, 1, -1])

    def __call__(self,):
        st = time.time()
        self.opt.step(lambda v:cost(v, self.circuit, self.feat, self.label, 0), self.param)
        return time.time() - st

class DNN:
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.model = MLP([in_dim, int(6*in_dim/(in_dim+2))])
        self.opt = torch.optim.SGD([{'params': self.model.parameters(), 'lr': 0.1, 'weight_decay': 5e-5}], momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self,):
        feat = torch.rand(4, self.in_dim)
        label = torch.randint(0, 2, (4,))
        st = time.time()
        predict_train = self.model(feat)
        loss = self.loss_fn(predict_train, label.long())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return time.time() - st

if __name__ == '__main__':
    qiskit.IBMQ.load_account()
    cfg = get_config('config/qnn/qnn_GD_4_0_0-wine.yml')
    time_it = {'QNNN,FT':[], 'QNNN,NISQ':[], 'MLP':[]}
    for n_qubits in tqdm(range(1, 21, 1), desc='n_qubit'):
        cfg.CIRCUIT.N_QUBIT = n_qubits
        cfg.CIRCUIT.BACKEND = 'default.qubit'        
        QNNN = QNN(cfg)
        time_it['QNNN,FT'].append(QNNN())

        cfg.CIRCUIT.BACKEND = 'qiskit.aer'
        QNNN_noisy = QNN(cfg)
        time_it['QNNN,NISQ'].append(QNNN_noisy())

        mlp = DNN(n_qubits)
        time_it['MLP'].append(mlp())
    
    with open('test/time_qubit.json', 'w') as f:
        json.dump(time_it, f)
    with open('test/time_qubit.json', 'r') as f:
        time_it = json.load(f)

    params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
    from matplotlib import rcParams
    rcParams.update(params)

    fig, ax = plt.subplots()
    colors = ['#9467bd', '#1f78b4', '#ff7f0e', '#2ca02c', '#8c564b', '#e377c2', '#bcbd22', '#7f7f7f', '#17becf', '#d62728']
    cnt = 0
    axins = inset_axes(ax, width='40%', height='40%', loc='lower left', bbox_to_anchor=(0.2, 0.2, 1, 1), bbox_transform=ax.transAxes)
    for key in time_it:
        ax.plot(range(1, 21, 1), time_it[key], label=key, color=colors[cnt])
        ax.set_xticks(range(1, 21, 1))

        axins.plot(range(1, 21, 1), time_it[key], label=key, color=colors[cnt])
        axins.tick_params(labelsize=15)
        cnt += 1
    ax.legend(loc='best', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('Number of qubits', fontsize=20)
    ax.set_ylabel('Time s/iter', fontsize=20)

    axins.set_xlim(1, 20)
    axins.set_ylim(-0.1, 1)
    mark_inset(ax, axins, loc1a=3, loc1b=2, loc2a=4, loc2b=1, fc="none", ec='k', lw=1, linestyle='--')

    plt.tight_layout()
    plt.savefig('test/time_qubit.pdf', dpi=600, format='pdf')
    plt.show()