#encoding=utf-8
"""
Implementation of feature embedding circuit
--------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import pennylane as qml

def CNOT_layer(n_qubits=2):
    for i in range(0, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])
    for i in range(1, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])

def feature_embedding(data):
    '''
    embed classical data to quantum circuit
    --------------------------------------
    :param data: arrays, [n_layers, n_qubits]
    '''
    n_layers, n_qubits = data.shape[0], data.shape[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.RY(data[i, j], wires=j)
        CNOT_layer(n_qubits)

def feature_embedding_trainable(param, data=None):
    '''
    embed classical data to quantum circuit
    --------------------------------------
    :param data: arrays, [n_layers, n_qubits]
    :param param: arrays, [2*n_layers, n_qubits]
    '''
    n_layers, n_qubits = data.shape[0], data.shape[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.RX(data[i, j], wires=j)
        if i < n_layers - 1:
            param_index = 0
            for j in range(0, n_qubits, 2):
                qml.MultiRZ(param[i*2, param_index], wires=[j, j+1])
                param_index += 1
            for j in range(1, n_qubits, 2):
                if j+1 < n_qubits:
                    qml.MultiRZ(param[i*2, param_index], wires=[j, j+1])
                    param_index += 1
            qml.MultiRZ(param[i*2, param_index], wires=[0, n_qubits-1])
            for j in range(n_qubits):
                qml.RY(param[i*2+1, j], wires=j)
        else:
            CNOT_layer(n_qubits)