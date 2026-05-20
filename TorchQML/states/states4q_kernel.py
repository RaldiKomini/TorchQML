
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_per(specs):#
    x = Xsym()
    t = Tsym()

    num_qubits = 4
    num_blocks = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # Symmetry-adapted initialization (improves kernel positive-definiteness)
    c.add_gates([H, H, H, H])
   # c.add_gates([RZ(math.pi/4), I, RZ(math.pi/4), I])  # Shifted RZ for better gradients
    c.add_gates([X, I, X, I])

    t_idx = 0

    # Initial entangling layer (adds expressivity early)
    CRY(c, math.pi * t[t_idx], num_qubits, 0, 1); t_idx += 1
    CRY(c, math.pi * t[t_idx], num_qubits, 2, 3); t_idx += 1

    # Enhanced data reuploading with double encoding per block
    for block in range(num_blocks):
        for repeat in range(2):  # Double reupload per block for richer features [web:5]
            for q in range(num_qubits):
                i1 = (block * num_qubits + q) % xlen
                i2 = (block * num_qubits + q + 31) % xlen

                # Shifted angles prevent vanishing gradients
                angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1] + 0.1)
                angle_z = math.pi * (t[t_idx + 2] * x[i2] + t[t_idx + 3] + 0.1)
                t_idx += 4

                ry_layer = [I, I, I, I]
                rz_layer = [I, I, I, I]
                ry_layer[q] = ry(angle_y)
                rz_layer[q] = rz(angle_z)
                c.add_gates(ry_layer)
                c.add_gates(rz_layer)

        # Alternating entanglement with full ring + diagonals (better mixing than skip) [web:1]
        if block % 2 == 0:
            for q in range(num_qubits):
                ent = math.pi/4 * t[t_idx]  # Scaled for stability
                t_idx += 1
                CRY(c, ent, num_qubits, q, (q + 1) % num_qubits)
        else:
            # Add diagonal CNOT-like for long-range correlations
            #CZ(c, num_qubits, 0, 2); CY(c, num_qubits, 1, 3)  # Replace CRY with CZ for phase info
            CRY(c, math.pi/4 * t[t_idx], num_qubits, 0, 3); t_idx += 1
            CRY(c, math.pi/4 * t[t_idx], num_qubits, 1, 2); t_idx += 1


        c.add_gates([ry(math.pi/2), ry(math.pi/2), ry(math.pi/2), ry(math.pi/2)])

    return c





def state4_gp(specs):
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    num_blocks = 3  # keep small; kernel training likes shallow-ish maps

    c = Circuit(num_qubits=nq)

    # Start in uniform superposition
    c.add_gates([H]*nq)

    t_idx = 0

    # Shared per-qubit feature scalers (trainable)
    # a[q] multiplies the encoded feature for qubit q
    a = [t[t_idx+i] for i in range(nq)]
    t_idx += nq

    # Shared per-qubit bias (trainable)
    b = [t[t_idx+i] for i in range(nq)]
    t_idx += nq

    for block in range(num_blocks):
        # ---- Data-dependent Z phases (commuting part) ----
        # Choose a "global" feature index for each qubit that cycles through xlen
        gates_rz = []
        for q in range(nq):
            i = (block*nq + q) % xlen
            ang = math.pi * (a[q] * x[i] + b[q])
            gates_rz.append(rz(ang))
        c.add_gates(gates_rz)

        # ---- Trainable entangling phases (IQP-ish) ----
        # Use CRZ-like behavior with CRY if that's what you have: controlled rotations as entanglers.
        # Shared entanglement strength per block (keeps params small, stable)
        ent = math.pi * t[t_idx]; t_idx += 1
        for q in range(nq):
            CRY(c, ent, nq, q, (q+1) % nq)

        # ---- Non-commuting mixing to make it nontrivial ----
        # A fixed H layer between blocks turns phase map into something richer
        c.add_gates([H]*nq)

    # Small trainable final local mixing (shared)
    # Helps adjust geometry without turning into a classifier
    for q in range(nq):
        ry_layer = [I]*nq
        ry_layer[q] = ry(math.pi * t[t_idx]); t_idx += 1
        c.add_gates(ry_layer)

    return c






def state4_mid():
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 4
    num_blocks = 4      # fewer blocks
    xlen = 128          # uses first 128 features

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry with some X's
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q % 2 == 0 else I)
    c.add_gates(init_gates)

    # --- Initial param-only layer (2 t per qubit) ---
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

    # --- Data reuploading blocks ---
    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []

        for q in range(num_qubits):
            # pick one feature index per (block, qubit)
            i = (block * num_qubits + q) % xlen

            # Ry: pi * (a * x[i] + b)
            angle_y = math.pi * (t[t_idx] * x[i] + t[t_idx + 1])
            t_idx += 2

            # Rz: pi * (c * x[i] + d)
            angle_z = math.pi * (t[t_idx] * x[i] + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)

        # --- Simple entangling: nearest-neighbor ring ---
        for q in range(num_qubits):
            q_next = (q + 1) % num_qubits
            c.add_full(CNOT(num_qubits, q, q_next))

    # Params:
    # - initial layer: 4 qubits * 2 = 8
    # - blocks: num_blocks * 4 qubits * 4 params = 4 * 4 * 4 = 64
    # Total nt = 8 + 64 = 72
    return c



def s4q1():
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 4
    xlen = 56
    num_blocks = 6   # data-reuploading blocks

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry a bit
    init = []
    for q in range(num_qubits):
        init.append(X if q in (0, 2) else I)
    c.add_gates(init)

    t_idx = 0

    # --- Initial param-only layer: 3 params / qubit (Ry, Rz, Rx) ---
    g_ry0, g_rz0, g_rx0 = [], [], []
    for q in range(num_qubits):
        ty = t[t_idx]; t_idx += 1
        tz = t[t_idx]; t_idx += 1
        tx = t[t_idx]; t_idx += 1

        g_ry0.append(ry(math.pi * ty))
        g_rz0.append(rz(math.pi * tz))
        g_rx0.append(rx(math.pi * tx))

    c.add_gates(g_ry0)
    c.add_gates(g_rz0)
    c.add_gates(g_rx0)

    # --- Data reuploading blocks ---
    for b in range(num_blocks):
        g_ry, g_rz, g_rx = [], [], []

        for q in range(num_qubits):
            base = (b * num_qubits + q) % xlen
            i1 = base
            i2 = (base + 13) % xlen
            i3 = (base + 29) % xlen

            # Ry: pi * (a1 * x[i1] + a2 * x[i2] + a0)
            ay1 = t[t_idx]; ay2 = t[t_idx + 1]; ay0 = t[t_idx + 2]; t_idx += 3
            angle_y = math.pi * (ay1 * x[i1] + ay2 * x[i2] + ay0)

            # Rz: pi * (b1 * (x[i2] + x[i3]) + b0)
            bz1 = t[t_idx]; bz0 = t[t_idx + 1]; t_idx += 2
            angle_z = math.pi * (bz1 * (x[i2] + x[i3]) + bz0)

            # Rx: pi * (c1 * (x[i1] - x[i3]) + c2 * x[i2] + c0)
            cx1 = t[t_idx]; cx2 = t[t_idx + 1]; cx0 = t[t_idx + 2]; t_idx += 3
            angle_x = math.pi * (cx1 * (x[i1] - x[i3]) + cx2 * x[i2] + cx0)

            g_ry.append(ry(angle_y))
            g_rz.append(rz(angle_z))
            g_rx.append(rx(angle_x))

        c.add_gates(g_ry)
        c.add_gates(g_rz)
        c.add_gates(g_rx)

        # --- Entangling pattern per block ---
        if b % 3 == 0:
            # ring
            pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif b % 3 == 1:
            # cross pairs
            pairs = [(0, 2), (1, 3)]
        else:
            # star from qubit 0
            pairs = [(0, 1), (0, 2), (0, 3)]

        for ctr, trg in pairs:
            c.add_full(CNOT(num_qubits, ctr, trg))

    return c



def state4_p2(specs):
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    num_blocks = 3

    c = Circuit(num_qubits=nq)
    c.add_gates([H]*nq)

    t_idx = 0

    # Shared scalers for (x_i, x_j, x_i*x_j) terms per qubit
    # Each qubit gets 3 scalers, shared across blocks
    A = [(t[t_idx+3*q+0], t[t_idx+3*q+1], t[t_idx+3*q+2]) for q in range(nq)]
    t_idx += 3*nq

    # Shared bias per qubit
    B = [t[t_idx+i] for i in range(nq)]
    t_idx += nq

    for block in range(num_blocks):
        for q in range(nq):
            k = block*nq + q
            i = k % xlen
            j = (k + xlen//2) % xlen  # opposite side; works for any xlen>=2

            a1, a2, a12 = A[q]
            # bilinear angle
            ang = math.pi * (a1*x[i] + a2*x[j] + a12*(x[i]*x[j]) + B[q])

            layer = [I]*nq
            # Use RY for non-commuting with entanglers
            layer[q] = ry(ang)
            c.add_gates(layer)

        # Trainable ring entanglement per block (shared)
        ent = math.pi * t[t_idx]; t_idx += 1
        for q in range(nq):
            CRY(c, ent, nq, q, (q+1) % nq)

        # fixed mixer
        c.add_gates([H]*nq)

    return c

__all__ = ['state4_per', 'state4_gp', 'state4_mid', 's4q1', 'state4_p2']
