
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_workhorse(specs):
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen

    feats_per_qubit = 4
    feats_per_block = nq * feats_per_qubit  # 16
    num_blocks = (xlen + feats_per_block - 1) // feats_per_block  # ceil

    c = Circuit(num_qubits=nq)

    # small symmetry break helps (optional)
    c.add_gates([H] * nq)
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])

    t_idx = 0

    for b in range(num_blocks):
        # ---- data reuploading block ----
        for q in range(nq):
            # 4 features per qubit per block (covers all x over blocks)
            base = b * feats_per_block + q * feats_per_qubit
            i0 = (base + 0) % xlen
            i1 = (base + 1) % xlen
            i2 = (base + 2) % xlen
            i3 = (base + 3) % xlen

            # pooled mixed scalar (keeps it stable and mixes signs)
            u = 0.25 * (x[i0] + x[i1] - x[i2] + x[i3])

            # Rx: pi * (a*u + b)
            ax = t[t_idx + 0]; bx = t[t_idx + 1]; t_idx += 2
            # Ry
            ay = t[t_idx + 0]; by = t[t_idx + 1]; t_idx += 2
            # Rz
            az = t[t_idx + 0]; bz = t[t_idx + 1]; t_idx += 2

            c.add_gates([rx(math.pi * (ax * u + bx)) if k == q else I for k in range(nq)])
            c.add_gates([ry(math.pi * (ay * u + by)) if k == q else I for k in range(nq)])
            c.add_gates([rz(math.pi * (az * u + bz)) if k == q else I for k in range(nq)])

        # ---- entanglement (alternate patterns) ----
        # ring
        for q in range(nq):
            c.add_full(CNOT(nq, q, (q + 1) % nq))

        # cross (alternate blocks)
        if (b % 2) == 1:
            c.add_full(CNOT(nq, 0, 2))
            c.add_full(CNOT(nq, 1, 3))

    # tlen = num_blocks * nq * 6 = num_blocks * 24
    # for xlen=128 -> 8 blocks -> 192 params
    return c




def state4_optimized(specs):
    x = Xsym()
    t = Tsym()

    num_qubits = 4
    num_blocks = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # Initial H + symmetry breaking
    c.add_gates([H, H, H, H])
    c.add_gates([X, I, X, I])

    # Initial layer (8 params)
    t_idx = 0
    for q in range(num_qubits):
        ry_layer = [I, I, I, I]
        rz_layer = [I, I, I, I]

        ry_layer[q] = ry(math.pi * t[t_idx])
        rz_layer[q] = rz(math.pi * t[t_idx + 1])

        c.add_gates(ry_layer)
        c.add_gates(rz_layer)

        t_idx += 2

    # Data reuploading
    for block in range(num_blocks):
        for q in range(num_qubits):
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 31) % xlen

            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            angle_z = math.pi * (t[t_idx + 2] * x[i2] + t[t_idx + 3])
            t_idx += 4

            ry_layer = [I, I, I, I]
            rz_layer = [I, I, I, I]

            ry_layer[q] = ry(angle_y)
            rz_layer[q] = rz(angle_z)

            c.add_gates(ry_layer)
            c.add_gates(rz_layer)

        # Dynamic entanglement
        if block % 2 == 0:  # Ring
            for q in range(num_qubits):
                ent = math.pi * t[t_idx]
                t_idx += 1
                CRY(c, ent, num_qubits, q, (q + 1) % num_qubits)
        else:  # Skip
            CRY(c, math.pi * t[t_idx], num_qubits, 0, 2); t_idx += 1
            CRY(c, math.pi * t[t_idx], num_qubits, 1, 3); t_idx += 1

    return c


__all__ = ['state4_workhorse', 'state4_optimized']
