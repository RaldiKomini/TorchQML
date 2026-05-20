import math
import torch
from TorchQML import Circuit
from TorchQML import H
from TorchQML import rx, ry, rz
from TorchQML import Xsym, Tsym



def stateg(spec) -> torch.Tensor:

    x = Xsym()
    t = Tsym()

    c = Circuit(num_qubits=1)

    c.add_gates([H])

    c.add_gates([ry(t[0])])

    c.add_gates([rx(math.pi * x[0])])

    c.add_gates([ry(t[1])])

    c.add_gates([rx(math.pi * x[1])])

    c.add_gates([rz(t[2])])

    c.add_gates([ry(t[3])])

    c.add_gates([rx(0.5 * math.pi * x[0])])

    c.add_gates([ry(0.5 * math.pi * x[1])])

    c.add_gates([rz(t[4])])

    c.add_gates([ry(t[5])])

    c.add_gates([rx(t[6] * math.pi * (x[0] + x[1]))])



    return c


def stategc() -> Circuit:
    # Now x is 1D: x[0] = r^2
    x = Xsym()
    t = Tsym()  # parameters

    c = Circuit(num_qubits=1)
    c.add_gates([H])
    c.add_gates([ry(t[0] + t[1] * x[0])])   # rotate according to r^2
    return c

import math

def state2():
    x = Xsym()   # symbolic inputs (x[0], x[1], ...)
    t = Tsym()   # symbolic trainable params (t[0], t[1], ...)

    c = Circuit(num_qubits=1)

    # --- Initial "featureless" superposition ---
    c.add_gates([H])

    # ---------- Block 0: simple mixing ----------
    c.add_gates([ry(t[0])])
    c.add_gates([rz(math.pi * (x[0] + x[1]) * t[1])])
    c.add_gates([rx(math.pi * (x[0] - x[1]) * t[2])])

    # ---------- Block 1: data reuploading ----------
    c.add_gates([ry(t[3] * math.pi * x[0])])
    c.add_gates([rz(t[4] * math.pi * x[1])])
    c.add_gates([rx(t[5] * math.pi * (x[0] + x[1]))])

    # ---------- Block 2: higher-order terms ----------
    c.add_gates([ry(t[6] * math.pi * (x[0] * x[1]))])
    c.add_gates([rz(t[7] * math.pi * (x[0] * x[0] - x[1] * x[1]))])
    c.add_gates([rx(t[8] * math.pi * (x[0] + x[1]))])

    # ---------- Block 3: reuploading with skewed combos ----------
    c.add_gates([ry(t[9])])
    c.add_gates([rz(t[10] * math.pi * (x[0] + 0.5 * x[1]))])
    c.add_gates([rx(t[11] * math.pi * (0.5 * x[0] - x[1]))])

    # ---------- Block 4: heavily mixed nonlinear features ----------
    c.add_gates([ry(t[12] * math.pi * (x[0] + x[1] + x[0] * x[1]))])
    c.add_gates([rz(t[13] * math.pi * (x[0] - x[1] + x[0] * x[1]))])
    c.add_gates([rx(t[14] * math.pi * (x[0] * x[0] + x[1] * x[1]))])

    # ---------- Final trainable rotation ----------
    c.add_gates([ry(t[15])])

    return c


def states() -> torch.Tensor:
    x = Xsym()   # image features: x[0], x[1], x[2], x[3]
    t = Tsym()   # trainable parameters t[0]..t[6]

    c = Circuit(num_qubits=1)

    c.add_gates([H])

    c.add_gates([ry(math.pi * x[0])])
    c.add_gates([rx(math.pi * x[1])])
    c.add_gates([rz(math.pi * x[2])])
    c.add_gates([ry(math.pi * x[3])])


    c.add_gates([rz(t[0])])
    c.add_gates([ry(t[1])])

    c.add_gates([rx(t[2] * (x[0] + x[1]))])
    c.add_gates([ry(t[3] * (x[2] + x[3]))])

    c.add_gates([rz(t[4])])
    c.add_gates([ry(t[5])])
    c.add_gates([rx(t[6])])

    return c


def state2_32():
    x = Xsym()   # x[0] ... x[31]
    t = Tsym()   # t[0], t[1], ...

    c = Circuit(num_qubits=1)

    # --- Initial superposition ---
    c.add_gates([H])

    k = 0  # index into t[]

    # ---------- Block 0: global mixing over all features ----------
    c.add_gates([ry(t[k])]); k += 1

    # sum_all = x[0] + x[1] + ... + x[31]
    sum_all = x[0]
    for i in range(1, 32):
        sum_all = sum_all + x[i]

    c.add_gates([rz(math.pi * sum_all * t[k])]); k += 1
    c.add_gates([rx(math.pi * (x[0] - x[1]) * t[k])]); k += 1  # keep a simple contrast term

    # ---------- Block 1: data reuploading for each feature ----------
    # One trainable ry per feature
    for i in range(32):
        c.add_gates([ry(t[k] * math.pi * x[i])])
        k += 1

    # ---------- Block 2: a few higher-order/global terms ----------
    # (uses simple aggregates so you don't have to manually write 32^2 terms)

    # mean feature
    mean_x = sum_all / 32.0

    # even vs odd index sums
    even_sum = x[0]
    for i in range(2, 32, 2):
        even_sum = even_sum + x[i]

    odd_sum = x[1]
    for i in range(3, 32, 2):
        odd_sum = odd_sum + x[i]

    c.add_gates([ry(t[k] * math.pi * (mean_x))]); k += 1
    c.add_gates([rz(t[k] * math.pi * (even_sum - odd_sum))]); k += 1
    c.add_gates([rx(t[k] * math.pi * (even_sum + odd_sum))]); k += 1

    # ---------- Final trainable rotation ----------
    c.add_gates([ry(t[k])]); k += 1

    return c



def statebr2():
    x = Xsym()     # symbolic inputs
    t = Tsym()     # symbolic trainable parameters

    xlen = 56
    tlen = 10
    c = Circuit(num_qubits=1)

    # initialize
    c.add_gates([H])

    # Feature encoding (same structure as your original code)
    for i in range(xlen):
        t_mod = t[i % tlen]

        c.add_gates([rz(t_mod)])                   # b.Rotz(t_mod)
        c.add_gates([rx(math.pi * x[i])])          # b.Rotx(pi * feature)
        c.add_gates([ry(math.pi * x[i])])          # b.Roty(pi * feature)
        c.add_gates([rz(math.pi * x[i])])          # b.Rotz(pi * feature)

    # Additional expressivity block
    for k in range(tlen):
        c.add_gates([rx(t[k])])
        c.add_gates([ry(t[k])])
        c.add_gates([rz(t[k])])

    return c



def s156():
    """
    1-qubit variational circuit for 56-dim input features.
    - Uses heavy data reuploading.
    - Exactly 10 trainable parameters (t[0]..t[9]), reused cyclically.

    Intended to be used with nt = 10.
    """

    x = Xsym()   # symbolic inputs: x[0]..x[55]
    t = Tsym()   # symbolic params: t[0]..t[9] (we will reuse them)

    c = Circuit(num_qubits=1)

    # Initial superposition
    c.add_gates([H])

    lenx = 56
    nt   = 10
    p    = 0

    def tp():
        """Return next parameter symbol t[p % nt] and advance p."""
        nonlocal p
        idx = p % nt
        p += 1
        return t[idx]

    # Number of reuploading blocks
    n_blocks = 12  # you can tweak this if you want more/less depth

    for b in range(n_blocks):
        # Pick two features for this block
        i0 = (2 * b) % lenx
        i1 = (2 * b + 1) % lenx

        # --- Data + param reuploading on 1 qubit ---
        # RY with param * feature
        c.add_gates([
            ry(math.pi * x[i0] * tp() * (-1))
        ])

        # RZ with param * other feature
        c.add_gates([
            rz(math.pi * x[i1] * tp())
        ])

        # RX with param * (sum of features)
        c.add_gates([
            rx(math.pi * (x[i0] + x[i1]) * tp())
        ])

    # Final trainable rotation (no data, just a free angle)
    c.add_gates([
        ry(tp())
    ])

    return c
