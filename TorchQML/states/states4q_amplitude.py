
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_amp(specs, num_blocks=4):#20 *nb
    x = Xsym()
    t = Tsym()

    nq = 4
    c = Circuit(num_qubits=nq)

    t_idx = 0
    for b in range(num_blocks):
        # --- Full SU(2) per qubit: Rz - Ry - Rz ---
        g_rz1, g_ry, g_rz2 = [], [], []
        for q in range(nq):
            g_rz1.append(rz(math.pi * t[t_idx])); t_idx += 1
            g_ry.append( ry(math.pi * t[t_idx])); t_idx += 1
            g_rz2.append(rz(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(g_rz1)
        c.add_gates(g_ry)
        c.add_gates(g_rz2)

        # --- Entanglers: alternate patterns ---
        if b % 2 == 0:
            c.add_full(CNOT(nq, 0, 1))
            c.add_full(CNOT(nq, 1, 2))
            c.add_full(CNOT(nq, 2, 3))
            c.add_full(CNOT(nq, 3, 0))
        else:
            c.add_full(CNOT(nq, 1, 0))
            c.add_full(CNOT(nq, 2, 1))
            c.add_full(CNOT(nq, 3, 2))
            c.add_full(CNOT(nq, 0, 3))
            c.add_full(CNOT(nq, 0, 2))
            c.add_full(CNOT(nq, 1, 3))

        # --- Light extra mixing (Rx + Rz) ---
        g_rx, g_rz = [], []
        for q in range(nq):
            g_rx.append(rx(math.pi * t[t_idx])); t_idx += 1
            g_rz.append(rz(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(g_rx)
        c.add_gates(g_rz)

    return c


def state4_amp2(specs, num_blocks=3):
    x = Xsym()
    t = Tsym()
    nq = 4
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)
    t_idx = 0

    for b in range(num_blocks):
        # trainable SU2
        g1,g2,g3 = [],[],[]
        for q in range(nq):
            g1.append(rz(math.pi * t[t_idx])); t_idx += 1
            g2.append(ry(math.pi * t[t_idx])); t_idx += 1
            g3.append(rz(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(g1); c.add_gates(g2); c.add_gates(g3)

        # data reupload (2 features per qubit, affine)
        grz, gry = [], []
        for q in range(nq):
            i1 = (b*nq + q) % xlen
            i2 = (b*nq + q + 5) % xlen

            ang_z = math.pi * (t[t_idx] * x[i1] + t[t_idx+1]); t_idx += 2
            ang_y = math.pi * (t[t_idx] * x[i2] + t[t_idx+1]); t_idx += 2

            grz.append(rz(ang_z))
            gry.append(ry(ang_y))
        c.add_gates(grz); c.add_gates(gry)

        # entangle
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 1, 2))
        c.add_full(CNOT(nq, 2, 3))
        c.add_full(CNOT(nq, 3, 0))

    return c





__all__ = ['state4_amp', 'state4_amp2']
