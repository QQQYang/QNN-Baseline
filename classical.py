"""
Train and evalute classical MLP
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, GradientDescentOptimizer, AdamOptimizer, RMSPropOptimizer

from argparse import Namespace
import argparse
import math
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from circuit import classifier, classifier_embedding, MLP
from tools import AverageMeter
from data_gen import data_gen, get_data

def get_opt():
    """
    Get parameters passed by python script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", type=int, default = 0)
    opt = parser.parse_args()
    return opt

def mse_loss(predict, label):
    return np.mean((predict - label)**2)

def cross_entropy(predict, label):
    prob_pos = np.exp(predict) / (np.exp(predict) + np.exp(1 - predict))
    prob_neg = 1 - prob_pos
    prob = np.concatenate((prob_pos[:, np.newaxis], prob_neg[:, np.newaxis]), axis=-1)
    label_onehot = ((label + 1) / 2)[:, np.newaxis]
    label_onehot = np.concatenate((label_onehot, 1 - label_onehot), axis=-1)
    prob = np.sum(prob * label_onehot, axis=-1)
    return np.mean(-prob * np.log(prob))

def cost(param, circuit, feat, label):
    exp = [circuit(param, feat[i]) for i in range(len(feat))]
    return mse_loss(np.array(exp), label)

def accuracy(predicts, labels):
    assert len(predicts) == len(labels)
    return np.sum((np.sign(predicts)*labels+1)/2)/len(predicts)

if __name__=='__main__':
    args = Namespace(
        epoch_num =  100,     # number of iterations
        iter_test = 1,
        n_depth_classifier = 2*3-2,   # number of blocks
        batch_size = 32,   # batch size for updating
        lr= 0.1,  # learning rate for G
        decay = 1, # learning rate
        Adam_flag = False,    # If use Adam optimizer
        file_trained_name = 'trained_para',
    )

    option = get_opt()

    time_qubit_acc = []
    for qubit in range(10):
        # generate data
        # data_gen(qubit, seed=option.bias)

        # load data
        feat_train, label_train, feat_test, label_test = get_data('data/feat_random_qnn_16_'+str(qubit)+'.npy', 'data/label_random_qnn_16_'+str(qubit)+'.npy', name='synthetic', random_label=True)
        feat_train = np.array(np.reshape(feat_train, (len(feat_train), -1)), np.float32)
        feat_train = (feat_train - np.mean(feat_train, axis=0)) / np.std(feat_train, axis=0)
        label_train = np.array((1 - label_train)/2, np.float32)

        feat_test = np.array(np.reshape(feat_test, (len(feat_test), -1)), np.float32)
        feat_test = (feat_test - np.mean(feat_test, axis=0)) / np.std(feat_test, axis=0)
        label_test = np.array((1 - label_test)/2, np.float32)

        n_qubits = feat_train.shape[-1]

        # training
        meter = AverageMeter()

        model = MLP([n_qubits, 128]) # synthetic data
        # model = MLP([n_qubits, 5]) # wine data
        model.train()
        opt = torch.optim.SGD([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': 5e-5}], momentum=0.9)
        lr_schedule = torch.optim.lr_scheduler.StepLR(opt, 40, gamma=0.1, last_epoch=-1)
        loss_fn = nn.CrossEntropyLoss()

        st = time.time()
        acc_train_list, acc_test_list = [], []
        for i in range(args.epoch_num):
            index = np.random.permutation(len(feat_train))
            feat_train, label_train = feat_train[index], label_train[index]
            for j in range(len(feat_train)//args.batch_size):
                feat_train_batch = feat_train[j*args.batch_size:(j+1)*args.batch_size]
                label_train_batch = label_train[j*args.batch_size:(j+1)*args.batch_size]

                feat_train_batch = torch.from_numpy(feat_train_batch)
                label_train_batch = torch.from_numpy(label_train_batch)
                predict_train = model(feat_train_batch)
                loss = loss_fn(predict_train, label_train_batch.long())
                opt.zero_grad()
                loss.backward()
                opt.step()

                acc_train = torch.sum(torch.argmax(predict_train, 1) == label_train_batch).double().cpu().numpy() / args.batch_size
                meter.update(acc_train)
                print(
                    'Epoch: {:5d}, iter: {:5d}, train_acc: {:0.7f}, loss: {:0.3f}, lr: {:0.7f}'.format(i, j, meter.avg, loss.item(), opt.param_groups[0]['lr'])
                )
            model.eval()
            predict_train = model(torch.from_numpy(feat_train))
            acc_train_list.append(torch.sum(torch.argmax(predict_train, 1) == torch.from_numpy(label_train)).double().cpu().numpy() / len(feat_train))

            predict_test = model(torch.from_numpy(feat_test))
            acc_test_list.append(torch.sum(torch.argmax(predict_test, 1) == torch.from_numpy(label_test)).double().cpu().numpy() / len(feat_test))
            model.train()
            print(
                'Epoch: {:5d}, train_acc: {:0.7f}, test_acc: {:0.7f}'.format(i, acc_train, acc_test_list[-1])
            )
            lr_schedule.step()
        if not os.path.exists(os.path.join('logs/mlp', str(qubit))):
            os.makedirs(os.path.join('logs/mlp', str(qubit)))
        np.save(os.path.join('logs/mlp', str(qubit), 'acc_train_mlp_random-synthetic-plus'), np.array(acc_train_list))
        np.save(os.path.join('logs/mlp', str(qubit), 'acc_test_mlp_random-synthetic-plus'), np.array(acc_test_list))
    # print('Train acc = {:0.7f}, test acc = {:0.7f}'.format(np.mean(np.array(time_qubit_acc)[:, 2]), np.mean(np.array(time_qubit_acc)[:, 3])))

