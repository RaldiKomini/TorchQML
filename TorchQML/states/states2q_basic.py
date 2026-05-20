from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT
from TorchQML import Xsym, Tsym
import math






def statew():
    x = Xsym()    # symbolic features: x[0], x[1], x[2], ...
    t = Tsym()    # trainable params: t[0] ... t[14]

    c = Circuit(num_qubits=2)

    # we will feed the FULL image as a flattened vector:
    # 3 channels * 64 * 64 = 12288
    NFEAT = 64   # must match how you flatten Xtr

    # initial superposition on both qubits
    c.add_gates([H, H])

    # 15 theta -> 5 layers x 3 params per layer
    num_layers = 5

    for layer in range(num_layers):
        th0 = t[3*layer + 0]
        th1 = t[3*layer + 1]
        th2 = t[3*layer + 2]

        # pick two features from the flattened image (wrap over NFEAT)
        i0 = (2*layer)     % NFEAT
        i1 = (2*layer + 1) % NFEAT

        # nonlinear combos of those two pixels
        a0 = math.pi * (x[i0] + th0)
        a1 = math.pi * (x[i1] + th1)
        a2 = math.pi * (x[i0] * x[i1] * th2)

        # layer 1: RY on each qubit
        c.add_gates([
            ry(a0),   # qubit 0
            ry(a1)    # qubit 1
        ])

        # layer 2: correlated RZs
        c.add_gates([
            rz(a2),   # qubit 0
            rz(-a2)   # qubit 1
        ])

    return c



def statew2():
    x = Xsym()    # symbolic features: x[0], x[1], x[2], ...
    t = Tsym()    # trainable params: t[0] ... t[14]

    c = Circuit(num_qubits=2)

    # FULL image flattened: 3 * 64 * 64 = 12288
    NFEAT = 64

    # initial superposition
    c.add_gates([H, H])

    # 15 theta -> use them in 5 blocks
    num_blocks = 5

    for b in range(num_blocks):
        th0 = t[3*b + 0]
        th1 = t[3*b + 1]
        th2 = t[3*b + 2]

        # pick 4 different pixels per block (wrap over NFEAT)
        base = (8 * b) % NFEAT
        i0 = (base + 0) % NFEAT
        i1 = (base + 1) % NFEAT
        i2 = (base + 2) % NFEAT
        i3 = (base + 3) % NFEAT

        # local "patch" aggregates
        patch0 = x[i0] + x[i1]          # like a tiny 2-pixel patch
        patch1 = x[i2] + x[i3]
        diff   = patch0 - patch1
        cross  = x[i0] * x[i3] - x[i1] * x[i2]

        # some global-ish mixing using a few far pixels
        j0 = (base + 100) % NFEAT
        j1 = (base + 500) % NFEAT
        glob = x[j0] + x[j1]

        # angles - heavily mixed nonlinear combos
        a0 = math.pi * (th0 * patch0 + th2 * cross + 0.5 * glob)
        a1 = math.pi * (th1 * patch1 - th2 * cross - 0.5 * glob)

        b0 = math.pi * (th0 * diff + th1 * cross)
        b1 = math.pi * (th1 * diff - th0 * cross)

        c0 = math.pi * (th2 * (patch0 * patch1) + glob)
        c1 = math.pi * (th2 * (patch0 - patch1) * glob)

        # ---- block structure ----
        # 1) data reuploading via RY
        c.add_gates([
            ry(a0),   # qubit 0
            ry(a1)    # qubit 1
        ])

        # 2) correlated RZ
        c.add_gates([
            rz(b0),   # qubit 0
            rz(b1)    # qubit 1
        ])

        # 3) RX mixing with stronger nonlinear features
        c.add_gates([
            rx(c0),   # qubit 0
            rx(c1)    # qubit 1
        ])

    return c



    # in TorchQML/states/states2q.py



__all__ = ['statew', 'statew2']
