
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY
from TorchQML import Xsym, Tsym
import math



def state5_amp2(specs, num_blocks=3):
    x = Xsym()
    t = Tsym()
    nq = 5
    xlen = specs.xlen

    c = Circuit(num_qubits=nq)
    t_idx = 0

    for b in range(num_blocks):
        # trainable SU2
        g1, g2, g3 = [], [], []
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

        # entangle (ring)
        c.add_full(CNOT(nq, 0, 1))
        c.add_full(CNOT(nq, 1, 2))
        c.add_full(CNOT(nq, 2, 3))
        c.add_full(CNOT(nq, 3, 4))
        c.add_full(CNOT(nq, 4, 0))

    return c



def rq5(spec, n_layers=4):
    """
    5-qubit variational reuploading circuit.
    n_layers: number of data reuploading blocks

    Total parameters = 5 qubits * 3 gates * n_layers
                      + 5 qubits * 3 final layer
                      = 15*(n_layers+1)
    """
    x = Xsym()        # symbolic features: x[i]
    t = Tsym()        # symbolic trainable params: t[i]
    c = Circuit(num_qubits=5)

    # Initial superposition
    c.add_gates([H, H, H, H, H])

    lenx = 64     # total feature count
    p = 0             # index into trainable parameters

    for layer in range(n_layers):
        # ----- Feature Encoding -----
        # RX, RY per qubit with different features
        # spread input features over all 5 qubits
        idx = (5 * layer) % lenx
        c.add_gates([
            rx(math.pi * x[(idx + 0) % lenx]),
            rx(math.pi * x[(idx + 1) % lenx]),
            rx(math.pi * x[(idx + 2) % lenx]),
            rx(math.pi * x[(idx + 3) % lenx]),
            rx(math.pi * x[(idx + 4) % lenx]),
        ])
        c.add_gates([
            ry(math.pi * x[(idx + 0) % lenx]),
            ry(math.pi * x[(idx + 1) % lenx]),
            ry(math.pi * x[(idx + 2) % lenx]),
            ry(math.pi * x[(idx + 3) % lenx]),
            ry(math.pi * x[(idx + 4) % lenx]),
        ])

        # ----- Trainable single-qubit layers -----
        # RZ -> RY -> RX for each qubit (3 params per qubit = 15 per layer)
        c.add_gates([ rz(t[p+0]),  rz(t[p+1]),  rz(t[p+2]),  rz(t[p+3]),  rz(t[p+4]) ])
        c.add_gates([ ry(t[p+5]),  ry(t[p+6]),  ry(t[p+7]),  ry(t[p+8]),  ry(t[p+9]) ])
        c.add_gates([ rx(t[p+10]), rx(t[p+11]), rx(t[p+12]), rx(t[p+13]), rx(t[p+14]) ])
        p += 15

        # ----- Entangling layer -----
        # chain of CNOTs: 0->1->2->3->4
      #  c.add_full(CNOT(0,1))
      #  c.add_full(CNOT(1,2))
      #  c.add_full(CNOT(2,3))
      #  c.add_full(CNOT(3,4))

    # ----- Final trainable layer -----
    # 15 more parameters
    c.add_gates([ rz(t[p+0]),  rz(t[p+1]),  rz(t[p+2]),  rz(t[p+3]),  rz(t[p+4]) ])
    c.add_gates([ ry(t[p+5]),  ry(t[p+6]),  ry(t[p+7]),  ry(t[p+8]),  ry(t[p+9]) ])
    c.add_gates([ rx(t[p+10]), rx(t[p+11]), rx(t[p+12]), rx(t[p+13]), rx(t[p+14]) ])

    return c
