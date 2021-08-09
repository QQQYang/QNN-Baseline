"""
Implementation of QAS circuit
--------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import pennylane as qml
from .embedding import feature_embedding, CNOT_layer
from .build import CIRCUIT_REGISTRY

arch = [
    [qml.RY, qml.RY, qml.RX, qml.RZ, qml.RX, qml.RY, qml.RY, qml.RY, qml.RZ, qml.RX, qml.RZ, qml.RY, qml.RY],
    [qml.RY, qml.RZ, qml.RZ, qml.RX, qml.RX, qml.RY, qml.RZ, qml.RY, qml.RY, qml.RX, qml.RY, qml.RY, qml.RZ]
]

def classifier(param, feat=None):
    '''
    Implementation of classification circuit.
    -----------------------------------------
    :param param: learnable parameters of classifier, [n_layers, n_qubits, 3]
    :param feat: classical features, [n_layers, n_qubits]
    Return expectation value of Pauli-Z on qubit 1
    '''
    feature_embedding(feat)
    n_qubits = feat.shape[1]
    n_layers = len(arch)
    for i in range(n_layers):
        for j in range(n_qubits):
            arch[i][j](param[i, j, 0], wires=j)
        CNOT_layer(n_qubits)
    return qml.expval(qml.PauliZ(0))

@CIRCUIT_REGISTRY.register()
def build_qnn_qas(cfg):
    return classifier