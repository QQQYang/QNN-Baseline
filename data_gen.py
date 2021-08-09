from pennylane import numpy as np
import pennylane as qml
import qiskit
import qiskit.providers.aer.noise as noise
import os
from tqdm import tqdm
import cv2

import math
import sys
sys.path.append('.')

from circuit import classifier, classifier_embedding
from tools.get_MNIST import load

def data_gen(n_qubits, root, n_depth_classifier=2, index=None, random_label=True, backend=None, net=None):
    n_data = 1000
    n_depth_embedding = 2

    # if backend is None:
    #     dev = qml.device("default.qubit", wires=n_qubits)
    # elif backend == 'qiskit.aer':
    #     # qiskit.IBMQ.load_account()
    #     provider = qiskit.IBMQ.providers(group='open')[0]
    #     backend_ibm = provider.get_backend('ibmq_16_melbourne')    # ibmq_qasm_simulator
    #     noise_model = noise.NoiseModel.from_backend(backend_ibm)

    #     dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
    # else:
    dev = qml.device("default.qubit", wires=n_qubits)
    if net is None or net=='qnn':
        circuit = qml.QNode(classifier, dev)
    else:
        circuit = qml.QNode(classifier_embedding, dev)

    np.random.seed(index)
    while True:
        param = np.random.uniform(0, math.pi, n_qubits*n_depth_classifier*3)
        param = np.reshape(param, (n_depth_classifier, n_qubits, 3))

        data = np.random.uniform(0, 2*math.pi, n_data*n_depth_embedding*n_qubits)
        data = np.array(data, requires_grad=False)
        data = np.reshape(data, (n_data, n_depth_embedding, n_qubits))

        gap_low, gap_high = 0.15, 0.15
        data_save, label_save = [], []
        for i in tqdm(range(n_data), desc='data_gen'):
            exp = circuit(param, feat=data[i])

            if exp < 0.0 - gap_low:
                data_save.append(data[i])
                label_save.append(-1)
            elif exp > 0.0 + gap_high:
                data_save.append(data[i])
                label_save.append(1)
        if sum(np.array(label_save) == 1) >= 200 and sum(np.array(label_save) == -1) >= 200:
            break
    print(sum(label_save))
    label_save = np.array(label_save)
    data_save = np.array(data_save)
    dataset_type = '_true_'
    if random_label:
        np.random.seed(index)
        random_index = np.random.permutation(len(label_save))
        label_save = label_save[random_index]
        dataset_type = '_random_'
    np.save(os.path.join(root, 'feat'+dataset_type+net+'_'+str(n_qubits)+'_'+str(index)), data_save)
    np.save(os.path.join(root, 'label'+dataset_type+net+'_'+str(n_qubits)+'_'+str(index)), label_save)

def get_data(feat_file=None, label_file=None, name='synthetic', random_label=True):
    if name == 'synthetic':
        feat = np.load(feat_file)
        label = np.load(label_file)

        index_neg = label==-1
        index_pos = label==1
        feat_pos_train = feat[index_pos][:100]
        feat_pos_test = feat[index_pos][100:200]
        feat_neg_train = feat[index_neg][:100]
        feat_neg_test = feat[index_neg][100:200]

        label_pos_train = label[index_pos][:100]
        label_pos_test = label[index_pos][100:200]
        label_neg_train = label[index_neg][:100]
        label_neg_test = label[index_neg][100:200]

        feat_train = np.array(np.concatenate((feat_pos_train, feat_neg_train), axis=0), requires_grad=False)
        label_train = np.array(np.concatenate((label_pos_train, label_neg_train), axis=0), requires_grad=False)

        feat_test = np.array(np.concatenate((feat_pos_test, feat_neg_test), axis=0), requires_grad=False)
        label_test = np.array(np.concatenate((label_pos_test, label_neg_test), axis=0), requires_grad=False)
        return feat_train, label_train, feat_test, label_test
    elif name == 'MNIST':
        feat_train, label_train, feat_test, label_test = load()
        for i in range(len(feat_train)):
            feat_train[i] = cv2.resize(feat_train[i], (10, 10), interpolation=cv2.INTER_NEAREST)
        for i in range(len(feat_test)):
            feat_test[i] = cv2.resize(feat_test[i], (10, 10), interpolation=cv2.INTER_NEAREST)
        # transform data from [0, 255] to [-1, 1]
        feat_train = feat_train / 255.0
        feat_test = feat_test / 255.0
        feat_train = math.pi*((feat_train - 0.1307) / 0.3081)
        feat_test = math.pi*((feat_test - 0.1307) / 0.3081)
        if random_label:
            index = np.random.permutation(len(label_train))
            label_train = label_train[index]
        return feat_train, label_train, feat_test, label_test
    elif name == 'WINE':
        feat, label = [], []
        with open(feat_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if float(parts[0]) > 2:
                    continue
                label.append(2*(float(parts[0])-1)-1)
                # label.append(float(parts[0])-1)
                feat.append([float(part) for part in parts[1:]])
        feat = np.array(feat, requires_grad=False)[:, np.newaxis, :]
        # feat = 2*math.pi*(feat - np.min(feat, axis=0)) / np.ptp(feat, axis=0)
        label = np.array(label, requires_grad=False)
        if random_label:
            index = np.random.permutation(len(label))
            label = label[index]

        index_neg = label==-1
        index_pos = label==1
        n_pos = sum(index_pos)
        n_neg = sum(index_neg)
        print('n_pos = {}, n_neg = {}'.format(n_pos, n_neg))
        feat_pos_train = feat[index_pos][:int(0.5*n_pos)]
        feat_pos_test = feat[index_pos][int(0.5*n_pos):]
        feat_neg_train = feat[index_neg][:int(0.5*n_neg)]
        feat_neg_test = feat[index_neg][int(0.5*n_neg):]

        label_pos_train = label[index_pos][:int(0.5*n_pos)]
        label_pos_test = label[index_pos][int(0.5*n_pos):]
        label_neg_train = label[index_neg][:int(0.5*n_neg)]
        label_neg_test = label[index_neg][int(0.5*n_neg):]

        feat_train = np.array(np.concatenate((feat_pos_train, feat_neg_train), axis=0), requires_grad=False)
        label_train = np.array(np.concatenate((label_pos_train, label_neg_train), axis=0), requires_grad=False)

        feat_test = np.array(np.concatenate((feat_pos_test, feat_neg_test), axis=0), requires_grad=False)
        label_test = np.array(np.concatenate((label_pos_test, label_neg_test), axis=0), requires_grad=False)
        return feat_train, label_train, feat_test, label_test

if __name__ == '__main__':
    # n_qubits = 10
    # n_data = 1000
    # n_depth_classifier = 2
    # n_depth_embedding = 2

    # dev = qml.device("default.qubit", wires=n_qubits)
    # circuit = qml.QNode(classifier, dev)

    # np.random.seed(0)
    # param = np.random.uniform(0, math.pi, n_qubits*n_depth_classifier*3)
    # param = np.reshape(param, (n_depth_classifier, n_qubits, 3))

    # data = np.random.uniform(0, 2*math.pi, n_data*n_depth_embedding*n_qubits)
    # data = np.array(data, requires_grad=False)
    # data = np.reshape(data, (n_data, n_depth_embedding, n_qubits))

    # gap_low, gap_high = 0.15, 0.15
    # data_save, label_save = [], []
    # for i in range(n_data):
    #     exp = circuit(param, data[i])

    #     if exp < 0.0 - gap_low:
    #         data_save.append(data[i])
    #         label_save.append(-1)
    #     elif exp > 0.0 + gap_high:
    #         data_save.append(data[i])
    #         label_save.append(1)
    # print(sum(label_save))
    # np.save('data/feat', np.array(data_save))
    # np.save('data/label', np.array(label_save))

    data_gen(16, 'data', 0)
