#encoding=utf-8
"""
Implementation of classification circuit
--------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import pennylane as qml
from .embedding import feature_embedding, CNOT_layer, feature_embedding_trainable
from .build import CIRCUIT_REGISTRY

def classifier(param, feat=None):
    '''
    Implementation of classification circuit.
    -----------------------------------------
    :param param: learnable parameters of classifier, [n_layers, n_qubits, 3]
    :param feat: classical features, [n_layers, n_qubits]
    Return expectation value of Pauli-Z on qubit 1
    '''
    feature_embedding(feat)
    n_layers = param.shape[0]
    n_qubits = feat.shape[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(param[i, j, 0], param[i, j, 1], param[i, j, 2], wires=j)
        CNOT_layer(n_qubits)
    return qml.expval(qml.PauliZ(0))

def classifier_embedding(param, feat=None):
    '''
    Implementation of classification circuit.
    -----------------------------------------
    :param param: learnable parameters of classifier, [3*n_layers-2, n_qubits, 3]
    :param feat: classical features, [n_layers, n_qubits]
    Return expectation value of Pauli-Z on qubit 1
    '''
    n_layers = feat.shape[0]
    n_qubits = feat.shape[1]
    feature_embedding_trainable(param[:2*n_layers, :, 0], data=feat)
    for i in range(2*n_layers, len(param)):
        for j in range(n_qubits):
            qml.Rot(param[i, j, 0], param[i, j, 1], param[i, j, 2], wires=j)
        CNOT_layer(n_qubits)
    return qml.expval(qml.PauliZ(0))

@CIRCUIT_REGISTRY.register()
def build_qnn(cfg):
    return classifier

@CIRCUIT_REGISTRY.register()
def build_embedding(cfg):
    return classifier_embedding