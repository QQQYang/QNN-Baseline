"""
Trainer for QNN
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
from sklearn.metrics import accuracy_score

import yaml
from easydict import EasyDict
import argparse
import os
import sys
import math
import time
from tqdm import tqdm

sys.path.append('.')

from circuit import build_circuit
from tools import AverageMeter
from data_gen import data_gen, get_data
from tools.metric import cost, accuracy, build_optimizer, build_scheduler
from tools.logger import setup_logger

class QuantumTrainer:
    def __init__(self, cfg, index=0, name='qnn', is_test=False):
        self.cfg = cfg
        if cfg.TRAIN.OPT == 'NGD':
            self.opt = QNGOptimizer(cfg.TRAIN.LR)
        else:
            self.opt = GradientDescentOptimizer(cfg.TRAIN.LR)
        net = build_circuit(cfg)
        if cfg.CIRCUIT.BACKEND == 'qiskit.aer':
            qiskit.IBMQ.load_account()
            provider = qiskit.IBMQ.providers(group='open')[0]
            backend = provider.get_backend('ibmq_16_melbourne')    # ibmq_qasm_simulator
            noise_model = noise.NoiseModel.from_backend(backend)

            dev = qml.device(cfg.CIRCUIT.BACKEND, wires=cfg.CIRCUIT.N_QUBIT, noise_model=noise_model, shots=10)
        else:
            dev = qml.device(cfg.CIRCUIT.BACKEND, wires=cfg.CIRCUIT.N_QUBIT)
        self.circuit = qml.QNode(net, dev)
        param = np.random.uniform(0, math.pi, cfg.CIRCUIT.N_QUBIT*cfg.CIRCUIT.N_DEPTH*3)
        self.param = np.reshape(param, (cfg.CIRCUIT.N_DEPTH, cfg.CIRCUIT.N_QUBIT, 3))
        if is_test:
            self.param = np.load(os.path.join(cfg.LOG_DIR, str(index), 'param_'+os.path.splitext(name.split('/')[-1])[0])+'.npy')

        # load dataset
        self.index = index
        if 'FEAT_FILE' in cfg.DATASET and os.path.exists(cfg.DATASET.FEAT_FILE):
            feat_path = '_'.join(cfg.DATASET.FEAT_FILE.split('_')[:-1]+[str(index)])+'.npy'
            label_path = '_'.join(cfg.DATASET.LABEL_FILE.split('_')[:-1]+[str(index)])+'.npy'
            self.feat_train, self.label_train, self.feat_test, self.label_test = get_data(feat_path, label_path, name=cfg.DATASET.NAME)
        elif cfg.DATASET.NAME == 'synthetic':
            dataset_type = '_true_'
            net_name = '_'.join(cfg.CIRCUIT.NAME.split('_')[1:])
            if cfg.DATASET.RANDOM_LABEL:
                dataset_type = '_random_'
            feat_path = os.path.join(cfg.DATASET.ROOT, 'feat'+dataset_type+net_name+'_'+str(cfg.CIRCUIT.N_QUBIT)+'_'+str(index)+'.npy')
            label_path = os.path.join(cfg.DATASET.ROOT, 'label'+dataset_type+net_name+'_'+str(cfg.CIRCUIT.N_QUBIT)+'_'+str(index)+'.npy')
            if is_test is False:
                if not os.path.exists(feat_path):
                    data_gen(n_qubits=cfg.CIRCUIT.N_QUBIT, root=cfg.DATASET.ROOT, n_depth_classifier=cfg.CIRCUIT.N_DEPTH, index=index, random_label=cfg.DATASET.RANDOM_LABEL, backend=cfg.CIRCUIT.BACKEND, net=net_name)
            self.feat_train, self.label_train, self.feat_test, self.label_test = get_data(feat_path, label_path, name=cfg.DATASET.NAME)
        else:
            self.feat_train, self.label_train, self.feat_test, self.label_test = get_data('data/wine.data', '', name=cfg.DATASET.NAME, random_label=cfg.DATASET.RANDOM_LABEL)
        self.meter = AverageMeter()

        self.logger = setup_logger(os.path.join(cfg.LOG_DIR, str(self.index)), distributed_rank=0, name='qnn')
        self.logger.info('Intialization')
        self.logger.info('--------------')
        self.logger.info('Train sample: {}, test sample: {}'.format(len(self.feat_train), len(self.feat_test)))
        self.name = name.split('/')[-1]

    def train(self):
        acc, loss, acc_test_list = [], [], []
        acc_best = 0
        self.loss = 0
        for i in tqdm(range(self.cfg.TRAIN.N_EPOCH), desc=self.name+'_'+str(self.index)):
            index = np.random.permutation(len(self.feat_train))
            feat_train, label_train = self.feat_train[index], self.label_train[index]
            for j in tqdm(range(max(len(self.feat_train)//self.cfg.TRAIN.BATCH_SIZE, 1)), desc='iter'):
                feat_train_batch = feat_train[j*self.cfg.TRAIN.BATCH_SIZE:(j+1)*self.cfg.TRAIN.BATCH_SIZE]
                label_train_batch = label_train[j*self.cfg.TRAIN.BATCH_SIZE:(j+1)*self.cfg.TRAIN.BATCH_SIZE]

                self.step(feat_train_batch, label_train_batch)
                self.logger.info('Epoch: {:5d}, iter: {:5d}, train_acc: {:0.7f}'.format(i, j, self.meter.avg))
            predict_train = []
            for k in range(len(feat_train)):
                pred = self.circuit(self.param, feat=feat_train[k])
                predict_train.append(pred)
            #predict_train = np.array([self.circuit(self.param, feat=feat_train[k]) for k in range(len(feat_train))])
            predict_train = np.array(predict_train)
            acc_train = accuracy(predict_train, label_train)
            acc.append(acc_train)
            loss.append(self.loss)
            np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'param_'+os.path.splitext(self.name)[0]), self.param)

            predict_test = np.array([self.circuit(self.param, feat=self.feat_test[k]) for k in range(len(self.feat_test))])
            acc_test = accuracy(predict_test, self.label_test)
            acc_best = max(acc_test, acc_best)
            acc_test_list.append(acc_test)
            if self.cfg.TRAIN.EARLY_STOP > 0:
                if len(acc_test_list) > self.cfg.TRAIN.EARLY_STOP and sum(np.array(acc_test_list)[-self.cfg.TRAIN.EARLY_STOP:] >= acc_best) == 0:
                    break
            np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_train_'+os.path.splitext(self.name)[0]), np.array(acc))
            np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_test_'+os.path.splitext(self.name)[0]), np.array(acc_test_list))
            np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'loss_'+os.path.splitext(self.name)[0]), np.array(loss))
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_train_'+os.path.splitext(self.name)[0]), np.array(acc))
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_test_'+os.path.splitext(self.name)[0]), np.array(acc_test_list))
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'loss_'+os.path.splitext(self.name)[0]), np.array(loss))

    def step(self, feat, label):
        if self.cfg.TRAIN.OPT == 'NGD':
            grads = 0
            for i in range(len(feat)):
                metric_tensor = qml.metric_tensor(self.circuit)(self.param, feat=feat[i])
                grad, _ = self.opt.compute_grad(lambda v: cost(v, self.circuit, [feat[i]], [label[i]], self.cfg.TRAIN.WEIGHT_DECAY), (self.param,), {})
                grad_flat = np.array(list(_flatten(grad)))
                x_flat = np.array(list(_flatten(self.param)))
                grads = grads + np.linalg.solve(metric_tensor, grad_flat)
            x_new_flat = np.array(list(_flatten(self.param))) - self.opt._stepsize * grads / len(feat)
            self.param = unflatten(x_new_flat, self.param)
        else:
            self.param, self.loss = self.opt.step_and_cost(lambda v: cost(v, self.circuit, feat, label, self.cfg.TRAIN.WEIGHT_DECAY), self.param)
        predict_train = []
        for k in range(len(feat)):
            pred = self.circuit(self.param, feat=feat[k])
            predict_train.append(pred)
        #predict_train = np.array([self.circuit(self.param, feat=feat[k]) for k in range(len(feat))])
        predict_train = np.array(predict_train)
        acc_train = accuracy(predict_train, label)
        self.meter.update(acc_train)

    def evaluate(self):
        pred = []
        for i in tqdm(range(len(self.feat_test)), desc='test'):
            pred.append(self.circuit(self.param, feat=self.feat_test[i]))
        acc_test = accuracy(np.array(pred), self.label_test)
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_test_'+os.path.splitext(self.name)[0]), np.array(acc_test))


class TorchTrainer:
    def __init__(self, cfg, index=0, name='qcnn', is_test=False):
        self.cfg = cfg
        self.net = build_circuit(cfg)
        self.opt = build_optimizer(cfg, self.net)
        self.scheduler = build_scheduler(cfg, self.opt)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if is_test:
            self.net.load_state_dict(torch.load(os.path.join(self.cfg.LOG_DIR, str(index), 'param_'+os.path.splitext(name.split('/')[-1])[0]+'.pth')))

        # load MNIST dataset
        dataset_type = '_true_'
        net_name = 'qnn'
        if cfg.DATASET.RANDOM_LABEL:
            dataset_type = '_random_'
        feat_path = os.path.join(cfg.DATASET.ROOT, 'feat'+dataset_type+net_name+'_'+str(cfg.CIRCUIT.N_QUBIT)+'_'+str(index)+'.npy')
        label_path = os.path.join(cfg.DATASET.ROOT, 'label'+dataset_type+net_name+'_'+str(cfg.CIRCUIT.N_QUBIT)+'_'+str(index)+'.npy')
        self.feat_train, self.label_train, self.feat_test, self.label_test = get_data(feat_file=feat_path, label_file=label_path, name=cfg.DATASET.NAME, random_label=cfg.DATASET.RANDOM_LABEL)
        if len(self.feat_train.shape) < 4:
            self.feat_train = np.reshape(self.feat_train, (-1, 1, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]))
            self.feat_test = np.reshape(self.feat_test, (-1, 1, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]))
            self.label_train = (self.label_train + 1)/2
            self.label_test = (self.label_test + 1)/2
        self.meter = AverageMeter()
        self.meter_acc = AverageMeter()

        self.index = index
        self.logger = setup_logger(os.path.join(cfg.LOG_DIR, str(self.index)), distributed_rank=0, name='qnn')
        self.logger.info('Intialization')
        self.name = name.split('/')[-1]

    def train(self):
        acc, loss = [], []
        acc_best = 0
        for i in tqdm(range(self.cfg.TRAIN.N_EPOCH), desc=self.name+'_'+str(self.index)):
            index = np.random.permutation(len(self.feat_train))
            #index = np.random.choice(index, self.cfg.DATASET.SAMPLE)
            index = index[:self.cfg.DATASET.SAMPLE]
            feat_train, label_train = self.feat_train[index], self.label_train[index]
            for j in tqdm(range(len(feat_train)//self.cfg.TRAIN.BATCH_SIZE), desc='iter'):
                feat_train_batch = feat_train[j*self.cfg.TRAIN.BATCH_SIZE:(j+1)*self.cfg.TRAIN.BATCH_SIZE]
                label_train_batch = label_train[j*self.cfg.TRAIN.BATCH_SIZE:(j+1)*self.cfg.TRAIN.BATCH_SIZE]

                loss_train, acc_train = self.step(torch.from_numpy(feat_train_batch).float(), torch.LongTensor(label_train_batch))
                self.meter.update(loss_train.item())
                self.meter_acc.update(acc_train)
                self.logger.info('Epoch: {:5d}, iter: {:5d}, loss: {:0.7f}'.format(i, j, self.meter.avg))

            self.scheduler.step()
            if i % self.cfg.TRAIN.EVAL_EPOCH == 0:
                self.net.eval()
                with torch.no_grad():
                    preds = []
                    for k in tqdm(range(len(self.feat_test)), desc='eval'):
                        pred = self.net(torch.from_numpy(self.feat_test[k]).float().unsqueeze(0))
                        preds.append(pred.argmax(-1).numpy())
                    acc_test = accuracy_score(self.label_test, np.array(preds))
                    self.logger.info('Epoch: {:5d}, test_acc: {:0.7f}'.format(i, acc_test))
                self.net.train()
            acc_best = max(self.meter_acc.avg, acc_best)
            acc.append(self.meter_acc.avg)
            loss.append(self.meter.avg)
            torch.save(self.net.cpu().state_dict(), os.path.join(self.cfg.LOG_DIR, str(self.index), 'param_'+os.path.splitext(self.name)[0]+'.pth'))
            if self.cfg.TRAIN.EARLY_STOP > 0:
                if len(acc) > self.cfg.TRAIN.EARLY_STOP and sum(np.array(acc)[-self.cfg.TRAIN.EARLY_STOP:] > acc_best) == 0:
                    break
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_train_'+os.path.splitext(self.name)[0]), np.array(acc))
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'loss_'+os.path.splitext(self.name)[0]), np.array(loss))

    def step(self, feat, label):
        pred = self.net(feat)
        loss = self.loss_fn(pred, label)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss, accuracy_score(label.numpy(), pred.argmax(-1).numpy())

    def evaluate(self):
        pred = []
        for i in tqdm(range(len(self.feat_test)), desc='test'):
            pred.append(self.net(torch.from_numpy(self.feat_test[i]).float().unsqueeze(0)))
        acc_test = accuracy_score(self.label_test, np.array(pred))
        np.save(os.path.join(self.cfg.LOG_DIR, str(self.index), 'acc_test_'+os.path.splitext(self.name)[0]), np.array(acc_test))
