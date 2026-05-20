from TorchQML import Circuit
from TorchQML.gates import H, X, I, Z, rx, ry, rz, CNOT
from TorchQML import Xsym, Tsym
import math


def block1(c, params, shift, entangle=True):
    """
    One block:
    - H
    - RX, RY, RZ on all qubits (12 params total for 4 qubits)
    - optional ring CNOT
    """
    nq = c.num_qubits
    i = shift

    # Basis change
    c.add_gates([H] * nq)

    # Single-qubit rotations
    c.add_gates([
        rx(params[i + 0]),
        rx(params[i + 1]),
        rx(params[i + 2]),
        rx(params[i + 3]),
    ])
    c.add_gates([
        ry(params[i + 4]),
        ry(params[i + 5]),
        ry(params[i + 6]),
        ry(params[i + 7]),
    ])
    c.add_gates([
        rz(params[i + 8]),
        rz(params[i + 9]),
        rz(params[i + 10]),
        rz(params[i + 11]),
    ])

    # Optional re-basis
    c.add_gates([H] * nq)

    # Optional entanglement
    if entangle:
        for q in range(nq):
            c.add_full(CNOT(nq, q, (q + 1) % nq))

    return c, shift + 12

def statel(specs):
    x = Xsym()
    t = Tsym()

    c = Circuit(num_qubits=specs.num_qubits)
    shift = 0

    # Alternate trainable and data blocks
    c, shift = block1(c, t, shift, entangle=True)
    c, shift = block1(c, x, shift, entangle=False)
    c, shift = block1(c, t, shift, entangle=True)
    c, shift = block1(c, x, shift, entangle=False)
    c, shift = block1(c, t, shift, entangle=True)

    return c
