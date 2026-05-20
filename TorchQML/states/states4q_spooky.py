
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_spooky_big(specs): #tlen 104
    """
    Entanglement-focused, high-capacity.
    Uses CRY/CRZ (and light local rotations) with data reupload.
    """
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    num_blocks = 6  # capacity knob

    c = Circuit(num_qubits=nq)

    # --- init: superposition + symmetry break ---
    c.add_gates([H] * nq)
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])

    t_idx = 0

    # --- initial local layer (small) ---
    # 2 params per qubit
    g1, g2 = [], []
    for q in range(nq):
        g1.append(ry(math.pi * t[t_idx])); t_idx += 1
        g2.append(rz(math.pi * t[t_idx])); t_idx += 1
    c.add_gates(g1)
    c.add_gates(g2)

    # Edge sets (directed as ctr->trg)
    ring = [(0,1), (1,2), (2,3), (3,0)]
    cross = [(0,2), (1,3)]
    all_edges = ring + cross  # 6 edges (dense for 4q)

    for b in range(num_blocks):
        # --- light local mixing per block (1 param per qubit) ---
        g = []
        for q in range(nq):
            g.append(rx(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(g)

        # choose feature indices for this block
        # (spread them so different edges see different x)
        base = (b * 3) % xlen

        # --- entangle: alternate CRY-heavy and CRZ-heavy blocks ---
        if b % 2 == 0:
            # CRY block: strong amplitude entanglement (good for your linear-entropy signal)
            for ei, (ctr, trg) in enumerate(all_edges):
                i = (base + 2*ei) % xlen
                ang = math.pi * (t[t_idx] * x[i] + t[t_idx + 1]); t_idx += 2
                CRY(c, ang, nq, ctr, trg)
        else:
            # CRZ block: phase correlations (adds capacity without wrecking everything)
            for ei, (ctr, trg) in enumerate(all_edges):
                i = (base + 2*ei + 1) % xlen
                ang = math.pi * (t[t_idx] * x[i] + t[t_idx + 1]); t_idx += 2
                CRZ(c, ang, nq, ctr, trg)

        # --- fixed entangling backbone (optional, adds stability) ---
        # (keeps some always-on coupling; remove if it over-entangles)
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 2, 3))

    return c



def state4_spooky01(specs, num_blocks=5):#125
    """
    Trainable-entanglement + data reuploading.
    Block structure (repeat):
        Local data+params  ->  Trainable entanglers  ->  Local params
    Designed for vector measurements (l_vec) + linear head.
    """
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    c = Circuit(num_qubits=nq)

    # --- init ---
    c.add_gates([H] * nq)
    c.add_gates([X if q % 2 == 0 else I for q in range(nq)])

    t_idx = 0

    # Entanglement topology (ring + one cross for capacity but not fully dense)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

    for b in range(num_blocks):
        # -------------------------
        # (A) Local data reupload (affine) : RY + RZ per qubit
        #     4 qubits * (2 params per gate) * 2 gates = 16 params/block
        # -------------------------
        gy, gz = [], []
        for q in range(nq):
            i = (b * nq + q) % xlen

            ang_y = math.pi * (t[t_idx] * x[i] + t[t_idx + 1]); t_idx += 2
            ang_z = math.pi * (t[t_idx] * x[i] + t[t_idx + 1]); t_idx += 2

            gy.append(ry(ang_y))
            gz.append(rz(ang_z))

        c.add_gates(gy)
        c.add_gates(gz)

        # -------------------------
        # (B) Trainable entanglement (parameter-only)
        #     1 param per edge (like learned coupling strengths)
        #     len(edges) params/block
        # -------------------------
        for (i, j) in edges:
            ang = math.pi * t[t_idx]; t_idx += 1
            # Use CRZ or CRY if that's what you have; RZZ is ideal if available.
            CRZ(c, ang, nq, i, j)

        # -------------------------
        # (C) Local mixing/readout alignment (parameter-only)
        #     RX per qubit: 4 params/block
        # -------------------------
        c.add_gates([rx(math.pi * t[t_idx + q]) for q in range(nq)])
        t_idx += nq

        # Optional: a light fixed backbone (keeps connectivity but not "trainable")
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 2, 3))

    # ---- exact parameter count ----
    # per block:
    #   (A) 16 + (B) len(edges)=5 + (C) 4  => 25 params/block
    # total tlen = 25 * num_blocks
    # e.g. num_blocks=5 -> tlen=125, num_blocks=6 -> tlen=150
    return c


def state4_spooky00(specs):#24
    """
    Trainable-entanglement spooky circuit.
    Optimized for vector-valued measurements + linear head.
    """
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)

    # -------------------------
    # 1) Initialization
    # -------------------------
    c.add_gates([H] * nq)
    c.add_gates([X if q % 2 == 0 else I for q in range(nq)])

    t_idx = 0

    # -------------------------
    # 2) First data injection (local, light)
    # -------------------------
    g = []
    for q in range(nq):
        i = q % xlen
        g.append(ry(math.pi * (t[t_idx] * x[i] + t[t_idx + 1])))
        t_idx += 2
    c.add_gates(g)

    # -------------------------
    # 3) Trainable entanglement layer
    # -------------------------
    # Ring with trainable strength (NO data)
    edges = [(0,1), (1,2), (2,3), (3,0)]
    for (ctr, trg) in edges:
        ang = math.pi * t[t_idx]
        t_idx += 1
        CRY(c, ang, nq, ctr, trg)

    # -------------------------
    # 4) Second data reupload (after entanglement)
    # -------------------------
    g = []
    for q in range(nq):
        i = (q + 5) % xlen
        g.append(rz(math.pi * (t[t_idx] * x[i] + t[t_idx + 1])))
        t_idx += 2
    c.add_gates(g)

    # -------------------------
    # 5) Readout alignment layer
    # -------------------------
    g = []
    for q in range(nq):
        g.append(ry(math.pi * t[t_idx]))
        t_idx += 1
    c.add_gates(g)

    # Optional light stabilizer
    c.add_full(CNOT(nq, 0, 2))
    c.add_full(CNOT(nq, 1, 3))

    return c


def state4_spooky0(specs):  # tlen = 104
    """
    Loss-optimized spooky circuit.
    Prioritizes stable per-qubit signals for spooky_loss + linear accuracy.
    """
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    num_blocks = 4   # fewer blocks = less chaos

    c = Circuit(num_qubits=nq)

    # --- init ---
    c.add_gates([H] * nq)
    c.add_gates([X if q % 2 == 0 else I for q in range(nq)])

    t_idx = 0

    # --- initial local alignment layer ---
    g1, g2 = [], []
    for q in range(nq):
        g1.append(ry(math.pi * t[t_idx])); t_idx += 1
        g2.append(rz(math.pi * t[t_idx])); t_idx += 1
    c.add_gates(g1)
    c.add_gates(g2)

    ring = [(0,1), (1,2), (2,3), (3,0)]

    for b in range(num_blocks):
        # --- data-driven local layer ---
        gy, gz = [], []
        for q in range(nq):
            i = (b * nq + q) % xlen
            gy.append(ry(math.pi * (t[t_idx] * x[i] + t[t_idx+1])))
            t_idx += 2
            gz.append(rz(math.pi * (t[t_idx] * x[i] + t[t_idx+1])))
            t_idx += 2

        c.add_gates(gy)
        c.add_gates(gz)

        # --- sparse entanglement ---
        for (ctr, trg) in ring:
            i = (b + ctr) % xlen
            ang = math.pi * (t[t_idx] * x[i] + t[t_idx+1])
            t_idx += 2
            CRY(c, ang, nq, ctr, trg)

        # --- light stabilizer ---
        c.add_full(CNOT(nq, 0, 2))
        c.add_full(CNOT(nq, 1, 3))

    return c


__all__ = ['state4_spooky_big', 'state4_spooky01', 'state4_spooky00', 'state4_spooky0']
