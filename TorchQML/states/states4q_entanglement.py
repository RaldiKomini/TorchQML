
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_entangle_improved(specs):#73
    x = Xsym()
    t = Tsym()

    nq = 4
    num_blocks = 3
    xlen = specs.xlen  # Assume >=12 for data cycling

    c = Circuit(num_qubits=nq)

    # --- Stronger initialization: H + global phase ---
    c.add_gates([H] * nq)
    c.add_gates([rz(math.pi * t[0])] * nq)
    t_idx = 1

    for b in range(num_blocks):
        # --- Layer 1: Data-modulated single-qubit rotations (stronger basis selection) ---
        for q in range(nq):
            angle = math.pi * (t[t_idx] * x[(b*4 + q) % xlen] + t[t_idx+1])
            layer = [I] * nq
            layer[q] = ry(angle)
            c.add_gates(layer)
            t_idx += 2

        # --- Layer 2: Backbone CNOT ladder (nearest + diagonal for GHZ-like) ---
        # Nearest neighbors
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 1, 2))
        c.add_full(CNOT(nq, 2, 3))
        c.add_full(CNOT(nq, 3, 0))

        # Skip connections (q0-q2, q1-q3)
        c.add_full(CNOT(nq, 0, 2))
        c.add_full(CNOT(nq, 1, 3))

        # --- Layer 3: Strong data-driven multi-controlled gates (CRX/CRZ mix) ---
        # Full pairwise CRX (amplitude entanglement)
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for ctrl, tgt in pairs:
            i = (b * 6 + pairs.index((ctrl,tgt))) % xlen
            angle = math.pi * (t[t_idx] * x[i] + t[t_idx+1])
            CRX(c, angle, nq, ctrl, tgt)  # CRX for real rotation mixing
            t_idx += 2

        # Diagonal CRZ (phase entanglement backbone)
        angle02 = math.pi * (t[t_idx] * x[(b*2) % xlen] + t[t_idx+1])
        angle13 = math.pi * (t[t_idx+2] * x[(b*2+1) % xlen] + t[t_idx+3])
        CRZ(c, angle02, nq, 0, 2)
        CRZ(c, angle13, nq, 1, 3)
        t_idx += 4

        # Optional: Measurement-induced mixing (if sim allows partial collapse)
        if b < num_blocks - 1:
            c.add_gates([H] * nq)  # Interference layer

    return c



def state4_entangle(specs):#tlen 48
    x = Xsym()
    t = Tsym()

    nq = 4
    num_blocks = 3   # shallow on purpose
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)

    # --- Initial superposition ---
    c.add_gates([H] * nq)

    t_idx = 0

    for b in range(num_blocks):

        # --- weak local rotations (just to break symmetry) ---
        gates = []
        for q in range(nq):
            angle = math.pi * t[t_idx]
            t_idx += 1
            gates.append(ry(angle))
        c.add_gates(gates)

        # --- entanglement-driven data injection ---
        # data controls HOW MUCH qubits entangle

        i = (b * nq) % xlen

        # ring entanglement with data-modulated strength
        for q in range(nq):
            qn = (q + 1) % nq
            angle = math.pi * (t[t_idx] * x[i] + t[t_idx + 1])
            t_idx += 2
            CRY(c, angle, nq, q, qn)

        # long-range entanglement
        angle = math.pi * (t[t_idx] * x[(i+3) % xlen] + t[t_idx + 1])
        t_idx += 2
        CRY(c, angle, nq, 0, 2)

        angle = math.pi * (t[t_idx] * x[(i+5) % xlen] + t[t_idx + 1])
        t_idx += 2
        CRY(c, angle, nq, 1, 3)

    return c


__all__ = ['state4_entangle_improved', 'state4_entangle']
