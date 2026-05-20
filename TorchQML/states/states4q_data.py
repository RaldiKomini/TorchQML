
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_ds1(specs, num_blocks=4):
    x = Xsym()
    t = Tsym()
    nq = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)
    t_idx = 0

    def on_q(g, q):
        layer = [I] * nq
        layer[q] = g
        return layer

    # Initial data encoding: use all features
    for q in range(nq):
        for f in range(4):
            feature_idx = q * 4 + f
            if feature_idx < xlen:
                angle = 2 * math.pi * x[feature_idx]
                c.add_gates(on_q(ry(angle), q))

    for b in range(num_blocks):
        # Layer 1: Single-qubit rotations
        for q in range(nq):
            c.add_gates(on_q(rz(math.pi * t[t_idx]), q)); t_idx += 1
            c.add_gates(on_q(ry(math.pi * t[t_idx]), q)); t_idx += 1
            c.add_gates(on_q(rz(math.pi * t[t_idx]), q)); t_idx += 1

        # Layer 2: Entanglement (full-dim CNOTs)
        for q in range(nq - 1):
            c.add_full(CNOT(nq, q, q + 1))
        c.add_full(CNOT(nq, nq - 1, 0))

        # Layer 3: More single-qubit rotations
        for q in range(nq):
            c.add_gates(on_q(rz(math.pi * t[t_idx]), q)); t_idx += 1
            c.add_gates(on_q(ry(math.pi * t[t_idx]), q)); t_idx += 1
            t_idx += 1  # Skip one for parameter count consistency

    return c





def state4_ds2(specs, num_blocks=3):
    x = Xsym()
    t = Tsym()
    nq = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)
    t_idx = 0

    def on_q(g, q):
        layer = [I] * nq
        layer[q] = g
        return layer

    for b in range(num_blocks):
        # Data re-uploading: encode features in each block
        for q in range(nq):
            f1 = (b * 2 * nq + 2*q) % xlen
            f2 = (b * 2 * nq + 2*q + 1) % xlen

            c.add_gates(on_q(ry(math.pi * x[f1]), q))
            c.add_gates(on_q(rz(math.pi * x[f2]), q))

        # Trainable unitaries
        for q in range(nq):
            c.add_gates(on_q(rz(math.pi * t[t_idx]), q)); t_idx += 1
            c.add_gates(on_q(ry(math.pi * t[t_idx]), q)); t_idx += 1
            c.add_gates(on_q(rz(math.pi * t[t_idx]), q)); t_idx += 1

        # Entanglement (except last block)
        if b < num_blocks - 1:
            for q in range(nq - 1):
                c.add_full(CNOT(nq, q, q + 1))

    return c


import math


def state4_ds3(specs, num_blocks=3):
    x = Xsym()
    t = Tsym()
    nq = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)
    t_idx = 0

    def on_q(g, q):
        layer = [I] * nq
        layer[q] = g
        return layer

    for b in range(num_blocks):
        # Trainable layers (act on current state, e.g., amplitude-encoded init)
        g1, g2, g3 = [], [], []
        for q in range(nq):
            g1.append(rz(math.pi * t[t_idx])); t_idx += 1
            g2.append(ry(math.pi * t[t_idx])); t_idx += 1
            g3.append(rz(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(g1)
        c.add_gates(g2)
        c.add_gates(g3)

        # Entangle
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 1, 2))
        c.add_full(CNOT(nq, 2, 3))
        c.add_full(CNOT(nq, 3, 0))

        # Additional data encoding (only in first 2 blocks)
        if b < 2:
            for q in range(nq):
                feature_idx = (b * nq + q) % xlen
                angle = math.pi * x[feature_idx]
                c.add_gates(on_q(ry(angle), q))

    return c



__all__ = ['state4_ds1', 'state4_ds2', 'state4_ds3']
