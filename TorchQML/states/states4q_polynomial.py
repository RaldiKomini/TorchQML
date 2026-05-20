
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_poly4x4(specs):#64
    x = Xsym()
    t = Tsym()

    nq = 4
    k = 4          # features per qubit per block
    deg = 4        # x^1..x^4
    xlen = specs.xlen

    feats_per_block = nq * k
    num_blocks = (xlen + feats_per_block - 1) // feats_per_block  # ceil

    c = Circuit(num_qubits=nq)

    # optional: start like your state4_big
    c.add_gates([H] * nq)
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])

    t_idx = 0

    for block in range(num_blocks):
        # For each of the k "slots" (0..3), apply one layer of Ry on all qubits.
        # Each qubit uses its own feature index for that slot.
        for s in range(k):
            gates = [I] * nq

            for q in range(nq):
                # feature indexing: sequential blocks of 16 features total
                # mapping: block*16 + q*k + s
                feat_idx = block * feats_per_block + q * k + s
                feat_idx = feat_idx % xlen  # keeps it flexible even if not divisible

                xi = x[feat_idx]

                # angle = pi * (a1*xi + a2*xi^2 + a3*xi^3 + a4*xi^4)
                a1 = t[t_idx + 0]
                a2 = t[t_idx + 1]
                a3 = t[t_idx + 2]
                a4 = t[t_idx + 3]
                t_idx += 4

                angle = math.pi * (a1 * xi + a2 * (xi * xi) + a3 * (xi * xi * xi) + a4 * (xi * xi * xi * xi))
                gates[q] = ry(angle)

            c.add_gates(gates)

        # entangle after each block (ring)
        for q in range(nq):
            c.add_full(CNOT(nq, q, (q + 1) % nq))

    # tlen = num_blocks * nq * k * deg
    # For xlen=128: num_blocks=8 => tlen=512
    return c

import math

import math


def state4_poly(specs):
    x = Xsym()
    t = Tsym()

    nq = 4
    k = 4
    xlen = specs.xlen

    feats_per_block = nq * k                      # 16
    num_blocks = (xlen + feats_per_block - 1) // feats_per_block  # ceil

    c = Circuit(num_qubits=nq)

    # start
    c.add_gates([H] * nq)
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])

    t_idx = 0

    for block in range(num_blocks):
        for s in range(k):
            gates_ry = [I] * nq
            gates_rz = [I] * nq

            for q in range(nq):
                feat_idx = (block * feats_per_block + q * k + s) % xlen
                xi = x[feat_idx]

                x2 = xi * xi
                x3 = x2 * xi
                x4 = x2 * x2

                # Ry coeffs (a0..a4)
                a0 = t[t_idx + 0]
                a1 = t[t_idx + 1]
                a2 = t[t_idx + 2]
                a3 = t[t_idx + 3]
                a4 = t[t_idx + 4]
                t_idx += 5

                ang_y = math.pi * (a0 + a1*xi + a2*x2 + a3*x3 + a4*x4)
                gates_ry[q] = ry(ang_y)

                # Rz coeffs (b0..b4)
                b0 = t[t_idx + 0]
                b1 = t[t_idx + 1]
                b2 = t[t_idx + 2]
                b3 = t[t_idx + 3]
                b4 = t[t_idx + 4]
                t_idx += 5

                ang_z = math.pi * (b0 + b1*xi + b2*x2 + b3*x3 + b4*x4)
                gates_rz[q] = rz(ang_z)

            c.add_gates(gates_ry)
            c.add_gates(gates_rz)

        # ring entanglement
        for q in range(nq):
            c.add_full(CNOT(nq, q, (q + 1) % nq))

    # tlen = num_blocks * feats_per_block * (2 axes) * (5 coeffs) = num_blocks * 160
    # for xlen=128: num_blocks=8 -> tlen=1280
    return c
import math


__all__ = ['state4_poly4x4', 'state4_poly']
