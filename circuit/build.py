# encoding: utf-8
"""
Build quantum circuit
------------------------------
Author: xxx
Email: xxx@xxx.com
"""

from tools.registry import Registry

CIRCUIT_REGISTRY = Registry("CIRCUIT")
CIRCUIT_REGISTRY.__doc__ = """
Registry for regression circuits.
"""

def build_circuit(cfg):
    """
    Build a circuit from `cfg.CIRCUIT.NAME`
    ----------------------------------
    Returns:
        an instance of circuit
    """

    circuit_name = cfg.CIRCUIT.NAME
    circuit = CIRCUIT_REGISTRY.get(circuit_name)(cfg)
    return circuit