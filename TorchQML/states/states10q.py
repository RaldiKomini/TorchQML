import torch
from TorchQML import Circuit
from TorchQML.gates.gate import H, X, I, Z
from TorchQML import rx, ry, rz, CNOT
from TorchQML import Sym, Xsym, Tsym

import math

# you can change this later to 16, 64, ...
xlen = 32





def state10_big():
    x = Xsym()   # symbolic input features: x[0], x[1], ...
    t = Tsym()   # symbolic parameters:    t[0], t[1], ...

    num_qubits = 10
    num_blocks = 4   # feature reupload blocks

    c = Circuit(num_qubits=num_qubits)

    # --- Initial bias + superposition ---
    # H on all qubits
    c.add_gates([H] * num_qubits)

    # Add some initial X to break symmetry
    init_gates = []
    for q in range(num_qubits):
        if q % 2 == 0:
            init_gates.append(X)
        else:
            init_gates.append(I)
    c.add_gates(init_gates)

    # --- Initial purely-parameterized layer (no x yet) ---
    t_idx = 0
    # For each qubit: Ry(pi * t[k]) then Rz(pi * t[k+1])
    gates_ry0 = []
    gates_rz0 = []
    for q in range(num_qubits):
        angle_y0 = math.pi * t[t_idx]; t_idx += 1
        angle_z0 = math.pi * t[t_idx]; t_idx += 1

        gates_ry0.append(ry(angle_y0))
        gates_rz0.append(rz(angle_z0))

    c.add_gates(gates_ry0)
    c.add_gates(gates_rz0)

    # --- Feature reuploading blocks ---
    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []
        gates_rx = []

        for q in range(num_qubits):
            # choose feature indices, wrapped mod xlen for flexibility
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 5) % xlen  # shifted index

            # --- Data-reuploading rotations ---
            # Ry:  angle = pi * ( a * x[i1] + b )
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2

            # Rz:  angle = pi * ( c * x[i2] + d )
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2

            # Rx:  angle = pi * ( e * (x[i1] + x[i2]) + f )
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        # apply three layers of single-qubit reuploading rotations
        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Strong entangling pattern ---
        if block % 2 == 0:
            # even blocks: nearest-neighbor ring, direction 0->1->2->...->9->0
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            # odd blocks: "skip" connections, 0->2->4->... plus a few cross links
            for q in range(0, num_qubits, 2):
                q_next = (q + 2) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))

            # extra long-range couplings
            c.add_full(CNOT(num_qubits, 0, 5))
            c.add_full(CNOT(num_qubits, 3, 8))

    return c
