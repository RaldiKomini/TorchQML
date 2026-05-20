from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT
from TorchQML import Xsym, Tsym
import math






def rh2():
    """
    2-qubit variational circuit for River vs Highway with ~56-dim input.

    - x: feature vector (56 dims from your image_to_features)
    - t: 18 trainable parameters (nt = 18)
    """
    x = Xsym()   # symbolic input features: x[0], x[1], ...
    t = Tsym()   # symbolic params: t[0], t[1], ...

    c = Circuit(num_qubits=2)

    # ---- Initial superposition ----
    c.add_gates([H, H])   # |++>

    p = 0      # parameter index in t
    L = 3      # number of blocks (depth)

    lenx = 56

    for layer in range(L):
        # pick 4 features for this layer (wrap around if needed)
        i0 = (4 * layer + 0) % lenx
        i1 = (4 * layer + 1) % lenx
        i2 = (4 * layer + 2) % lenx
        i3 = (4 * layer + 3) % lenx

        # ===== Data encoding layer =====
        # encode different features on each qubit
        c.add_gates([
            rx(math.pi * x[i0]),
            rx(math.pi * x[i1]),
        ])
        c.add_gates([
            ry(math.pi * x[i2]),
            ry(math.pi * x[i3]),
        ])

        # ===== Trainable single-qubit layer =====
        # z-rotations (per qubit)
        c.add_gates([
            rz(t[p + 0]),
            rz(t[p + 1]),
        ])
        # y-rotations (per qubit)
        c.add_gates([
            ry(t[p + 2]),
            ry(t[p + 3]),
        ])
        p += 4   # used 4 params in this block

        # ===== Entangling layer =====
        if layer % 2 == 0:
            # CNOT 0 -> 1
            c.add_full(CNOT(2, 0, 1))
       # else:
            # CNOT 1 -> 0 (or same direction if your CNOT is fixed)
        #    c.add_gates([CNOT])   # or a different gate if you have CX10 / CZ

    # Optional: final trainable layer (last 6 params)
    c.add_gates([
        rz(t[p + 0]),
        rz(t[p + 1]),
    ])
    c.add_gates([
        ry(t[p + 2]),
        ry(t[p + 3]),
    ])
    c.add_gates([
        rx(t[p + 4]),
        rx(t[p + 5]),
    ])
    # total params: p was 12 after 3 blocks, last layer uses 6 -> 18 total

    return c



# TorchQML/states/states2q.py



def rh22():
    """
    2-qubit variational circuit for River vs Highway
    designed for ~56-dim input features.

    Uses:
    - 4 data+param blocks
    - 30 parameters total (nt = 30)
    """

    x = Xsym()     # symbolic inputs: x[0], x[1], ...
    t = Tsym()     # symbolic params: t[0], t[1], ...

    c = Circuit(num_qubits=2)

    # Initial layer: both qubits in superposition
    c.add_gates([H, H])

    p = 0       # index into t
    n_blocks = 4

    lenx = 56
    for i in range(2):
        for layer in range(n_blocks):
            # ----- Data encoding -----
            # take 4 features per block, wrap around if needed
            i0 = (4 * layer + 0) % lenx
            i1 = (4 * layer + 1) % lenx
            i2 = (4 * layer + 2) % lenx
            i3 = (4 * layer + 3) % lenx

            # encode data on both qubits
            c.add_gates([
                rx(math.pi * x[i0]),
                rx(math.pi * x[i1]),
            ])
            c.add_gates([
                ry(math.pi * x[i2]),
                ry(math.pi * x[i3]),
            ])

            # ----- Trainable single-qubit layer -----
            # ZY on each qubit (4 params)
            c.add_gates([
                rz(t[p + 0]),
                rz(t[p + 1]),
            ])
            c.add_gates([
                ry(t[p + 2]),
                ry(t[p + 3]),
            ])
            # extra RX layer for flexibility (2 params)
            c.add_gates([
                rx(t[p + 4]),
                rx(t[p + 5]),
            ])
            p += 6   # 6 params per block

            # ----- Entangling layer -----
            c.add_full(CNOT(2, 0, 1))

        # Final trainable layer (6 more params)
        c.add_gates([
            rz(t[p + 0]),
            rz(t[p + 1]),
        ])
        c.add_gates([
            ry(t[p + 2]),
            ry(t[p + 3]),
        ])
        c.add_gates([
            rx(t[p + 4]),
            rx(t[p + 5]),
        ])
    # total params: p was 24 after 4 blocks, +6 = 30

    return c





def rh2q(nt: int,
                   depth: int = 4,
                   reuploads_per_block: int = 3,
                   lenx: int = 56) -> Circuit:
    """
    2-qubit reuploading ansatz for River vs Highway.

    - Input:  56-dim features (x[0]..x[55])
    - Qubits: 2
    - nt:     number of trainable parameters (you choose: 20, 50, 100, ...)
    - depth:  number of (data+param+entangler) blocks
    - reuploads_per_block: how many times to re-encode data within each block

    Parameters t[0..nt-1] are reused (wrapped) if the circuit asks for more.
    This lets you experiment with different nt without changing the ansatz.
    """

    x = Xsym()    # symbolic inputs
    t = Tsym()    # symbolic parameters

    c = Circuit(num_qubits=2)

    # Initial superposition
    c.add_gates([H, H])

    # helper to reuse parameters modulo nt
    param_idx = 0
    def next_param():
        nonlocal param_idx
        idx = param_idx % nt   # always 0..nt-1
        param_idx += 1
        return idx

    # main blocks
    for block in range(depth):
        # ----- DATA REUPLOAD LAYERS -----
        for r in range(reuploads_per_block):
            # pick 4 features for this (block, reupload) step
            base = (block * reuploads_per_block + r) * 4
            i0 = (base + 0) % lenx
            i1 = (base + 1) % lenx
            i2 = (base + 2) % lenx
            i3 = (base + 3) % lenx

            # encode on both qubits: RX/RY with different features
            c.add_gates([
                rx(math.pi   * x[i0]),
                rx(math.pi   * x[i1]),
            ])
            c.add_gates([
                ry(0.5*math.pi * x[i2]),
                ry(0.5*math.pi * x[i3]),
            ])

            # optional: extra small Z-encoding mixing pairs
            c.add_gates([
                rz(0.25*math.pi * (x[i0] + x[i2])),
                rz(0.25*math.pi * (x[i1] + x[i3])),
            ])

        # ----- TRAINABLE SINGLE-QUBIT LAYER -----
        # Each block gets a Z-Y-X trainable layer on each qubit (with parameter reuse)
        c.add_gates([
            rz(t[next_param()]),
            rz(t[next_param()]),
        ])
        c.add_gates([
            ry(t[next_param()]),
            ry(t[next_param()]),
        ])
        c.add_gates([
            rx(t[next_param()]),
            rx(t[next_param()]),
        ])

        # ----- ENTANGLING LAYER -----
        # Use a full 2-qubit CNOT each block
        c.add_full(CNOT(2, 0, 1))

    # Final trainable layer acts like a readout head.
    c.add_gates([
        rz(t[next_param()]),
        rz(t[next_param()]),
    ])
    c.add_gates([
        ry(t[next_param()]),
        ry(t[next_param()]),
    ])
    c.add_gates([
        rx(t[next_param()]),
        rx(t[next_param()]),
    ])

    return c


import math


__all__ = ['rh2', 'rh22', 'rh2q']
