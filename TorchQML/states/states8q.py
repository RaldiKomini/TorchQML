from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY
from TorchQML import Xsym, Tsym
import math


def state8_big(specs):
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params 208

    num_qubits = 8
    num_blocks = 4   # data-reuploading blocks

    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # --- Initial bias + superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry with some X's
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q % 2 == 0 else I)
    c.add_gates(init_gates)

    # --- Initial param-only layer ---
    t_idx = 0
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
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 5) % xlen

            # Ry: pi * (a * x[i1] + b)
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2

            # Rz: pi * (c * x[i2] + d)
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2

            # Rx: pi * (e * (x[i1] + x[i2]) + f)
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern ---
        if block % 2 == 0:
            # even blocks: nearest-neighbor ring
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            # odd blocks: skip connections + long range
            for q in range(0, num_qubits, 2):
                q_next = (q + 2) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))

            c.add_full(CNOT(num_qubits, 0, 4))
            c.add_full(CNOT(num_qubits, 3, 7))

    return c


def state8_feat56():
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 8
    xlen = 56           # number of features
    num_blocks = 5      # data-reuploading blocks

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry a bit
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q in (0, 3, 5) else I)
    c.add_gates(init_gates)

    t_idx = 0

    # --- Initial param-only layer: 3 params per qubit (Ry, Rz, Rx) ---
    gates_ry0, gates_rz0, gates_rx0 = [], [], []
    for q in range(num_qubits):
        ty = t[t_idx]; t_idx += 1
        tz = t[t_idx]; t_idx += 1
        tx = t[t_idx]; t_idx += 1

        gates_ry0.append(ry(math.pi * ty))
        gates_rz0.append(rz(math.pi * tz))
        gates_rx0.append(rx(math.pi * tx))

    c.add_gates(gates_ry0)
    c.add_gates(gates_rz0)
    c.add_gates(gates_rx0)

    # --- Data reuploading blocks ---
    for b in range(num_blocks):
        gates_ry, gates_rz, gates_rx = [], [], []

        for q in range(num_qubits):
            base = (b * num_qubits + q) % xlen
            i1 = base
            i2 = (base + 17) % xlen
            i3 = (base + 33) % xlen

            # Ry: pi * (a * x[i1] + b)
            ay1 = t[t_idx]; ay0 = t[t_idx + 1]; t_idx += 2
            angle_y = math.pi * (ay1 * x[i1] + ay0)

            # Rz: pi * (c * (x[i2] + x[i3]) + d)
            bz1 = t[t_idx]; bz0 = t[t_idx + 1]; t_idx += 2
            angle_z = math.pi * (bz1 * (x[i2] + x[i3]) + bz0)

            # Rx: pi * (e * (x[i1] - x[i2]) + f)
            cx1 = t[t_idx]; cx0 = t[t_idx + 1]; t_idx += 2
            angle_x = math.pi * (cx1 * (x[i1] - x[i2]) + cx0)

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern (varies by block) ---
        if b % 3 == 0:
            # ring: nearest neighbors
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        elif b % 3 == 1:
            # even/odd skip pairs
            pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
            for ctr, trg in pairs:
                c.add_full(CNOT(num_qubits, ctr, trg))
        else:
            # long-range connections
            pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
            for ctr, trg in pairs:
                c.add_full(CNOT(num_qubits, ctr, trg))

    return c
