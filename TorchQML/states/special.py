import math
from TorchQML.core.circuit import Circuit
from TorchQML import Xsym, Tsym
from TorchQML.gates import H, X, I, rx, ry, rz, CNOT  # adjust if your path differs


def big_tlen(num_qubits: int, num_blocks: int = 6) -> int:
    # initial: 2 params per qubit
    # per block: (Ry,Rz,Rx) each affine => 2 params each => 6 params per qubit per block
    return 2 * num_qubits + num_blocks * (6 * num_qubits)


def _state_big(
    specs,
    num_qubits: int,
    *,
    num_blocks: int = 6,
    offset: int = 7,
    assert_tlen: bool = False,
):
    """
    Generalized 'state4_big' style circuit.

    Structure:
      - H on all qubits
      - symmetry-breaking X on even qubits
      - initial param-only layer: Ry,Rz per qubit (2 params per qubit)
      - per-block data reupload: Ry,Rz,Rx per qubit, each affine in x (6 params per qubit per block)
      - entangling: even blocks = ring NN (only if nq>=2), odd blocks = long-range-ish pattern (safe for all nq)
    """
    x = Xsym()
    t = Tsym()
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # --- Break symmetry ---
    c.add_gates([X if (q % 2 == 0) else I for q in range(num_qubits)])

    # --- Initial param-only layer (2 t per qubit) ---
    t_idx = 0
    c.add_gates([ry(math.pi * t[t_idx + 2 * q]) for q in range(num_qubits)])
    c.add_gates([rz(math.pi * t[t_idx + 2 * q + 1]) for q in range(num_qubits)])
    t_idx += 2 * num_qubits

    # --- Data reuploading blocks ---
    for block in range(num_blocks):
        gates_ry, gates_rz, gates_rx = [], [], []

        for q in range(num_qubits):
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + offset) % xlen

            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1]); t_idx += 2
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1]); t_idx += 2
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1]); t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern (inside loop) ---
        # --- Entangling pattern ---
        if num_qubits == 1:
            pass
        elif block % 2 == 0:
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            if num_qubits == 2:
                c.add_full(CNOT(num_qubits, 0, 1))
            elif num_qubits == 3:
                c.add_full(CNOT(num_qubits, 0, 2))
                c.add_full(CNOT(num_qubits, 1, 2))
                c.add_full(CNOT(num_qubits, 0, 1))
            else:
                c.add_full(CNOT(num_qubits, 0, 2))
                c.add_full(CNOT(num_qubits, 1, 3))
                c.add_full(CNOT(num_qubits, 0, num_qubits - 1))
                if num_qubits >= 5:
                    c.add_full(CNOT(num_qubits, 2, 4))
                    c.add_full(CNOT(num_qubits, 1, 4))
    if assert_tlen:
        expected = big_tlen(num_qubits, num_blocks)
        assert t_idx == expected, f"t_idx={t_idx} expected={expected}"
        if hasattr(specs, "tlen"):
            assert specs.tlen == expected, f"specs.tlen={specs.tlen} expected={expected}"

    return c


# ----------------------------
# Public wrappers (state1_big ... state7_big)
# ----------------------------
def state1_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 1, num_blocks=num_blocks, offset=offset)

def state2_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 2, num_blocks=num_blocks, offset=offset)

def state3_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 3, num_blocks=num_blocks, offset=offset)

def state4_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 4, num_blocks=num_blocks, offset=offset)

def state5_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 5, num_blocks=num_blocks, offset=offset)

def state6_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 6, num_blocks=num_blocks, offset=offset)

def state7_big(specs, num_blocks: int = 6, offset: int = 7):
    return _state_big(specs, 7, num_blocks=num_blocks, offset=offset)
