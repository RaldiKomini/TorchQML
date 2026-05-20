import math

from TorchQML.gates import CNOT, H, I, rx, ry, rz


ROTATIONS = {"x": rx, "y": ry, "z": rz}


def rotation_layer(circ, idx_x, idx_t, x, t, *, axis: str = "x", mode: str = "same"):
    rot = ROTATIONS[axis]
    nq = circ.num_qubits

    if mode == "same":
        if idx_t is None or len(idx_t) != nq:
            raise ValueError("same-mode rotation needs one theta index per qubit")
        circ.add_gates([rot(math.pi * t[idx_t[q]]) for q in range(nq)])
        return circ

    if mode == "cycle":
        if idx_x is None or len(idx_x) != nq:
            raise ValueError("cycle-mode rotation needs one feature index per qubit")
        circ.add_gates([rot(math.pi * x[idx_x[q]]) for q in range(nq)])
        return circ

    if mode == "group":
        if idx_t is None or len(idx_t) != (nq + 1) // 2:
            raise ValueError("group-mode rotation needs one theta index per qubit pair")
        layer = [I] * nq
        group = 0
        for q in range(nq):
            layer[q] = rot(math.pi * t[idx_t[group]])
            if q % 2 == 1:
                group += 1
        circ.add_gates(layer)
        return circ

    raise ValueError(f"Unknown rotation mode: {mode}")


def ent_none(circ, idx_x, idx_t, x, t):
    return circ


def ent_chain(circ, idx_x, idx_t, x, t):
    for q in range(circ.num_qubits - 1):
        circ.add_full(CNOT(circ.num_qubits, q, q + 1))
    return circ


def ent_ring(circ, idx_x, idx_t, x, t):
    for q in range(circ.num_qubits):
        circ.add_full(CNOT(circ.num_qubits, q, (q + 1) % circ.num_qubits))
    return circ


def ent_pairs(circ, idx_x, idx_t, x, t):
    for q in range(0, circ.num_qubits, 2):
        if q + 1 < circ.num_qubits:
            circ.add_full(CNOT(circ.num_qubits, q, q + 1))
    return circ


def basis_all_h(circ, idx_x, idx_t, x, t):
    circ.add_gates([H] * circ.num_qubits)
    return circ


def basis_even_h(circ, idx_x, idx_t, x, t):
    layer = [I] * circ.num_qubits
    for q in range(0, circ.num_qubits, 2):
        layer[q] = H
    circ.add_gates(layer)
    return circ


def basis_odd_h(circ, idx_x, idx_t, x, t):
    layer = [I] * circ.num_qubits
    for q in range(1, circ.num_qubits, 2):
        layer[q] = H
    circ.add_gates(layer)
    return circ


def make_rot_pattern(axis: str, mode: str, nq: int):
    def pattern(circ, idx_x, idx_t, x, t):
        return rotation_layer(circ, idx_x, idx_t, x, t, axis=axis, mode=mode)

    pattern.name = f"rot_{axis}_{mode}"
    if mode == "same":
        pattern.kx = 0
        pattern.kt = nq
    elif mode == "cycle":
        pattern.kx = nq
        pattern.kt = 0
    elif mode == "group":
        pattern.kx = 0
        pattern.kt = (nq + 1) // 2
    else:
        raise ValueError(f"Unknown rotation mode: {mode}")
    return pattern


def mix_shape_basis_data(circ, idx_x, idx_t, x, t):
    nq = circ.num_qubits
    rotation_layer(circ, None, idx_t[:nq], x, t, axis="z", mode="same")
    basis_even_h(circ, None, None, x, t)
    rotation_layer(circ, idx_x[:nq], None, x, t, axis="x", mode="cycle")
    return circ


def mix_data_entangle(entangler):
    def pattern(circ, idx_x, idx_t, x, t):
        nq = circ.num_qubits
        rotation_layer(circ, idx_x[:nq], None, x, t, axis="x", mode="cycle")
        return entangler(circ, None, None, x, t)

    pattern.kx = lambda nq: nq
    pattern.kt = 0
    pattern.name = f"mix_data_{entangler.__name__}"
    return pattern


def macro_data_entangle_chain(circ, idx_x, idx_t, x, t):
    rotation_layer(circ, idx_x[: circ.num_qubits], None, x, t, axis="x", mode="cycle")
    return ent_chain(circ, None, None, x, t)


def macro_data_entangle_pairs(circ, idx_x, idx_t, x, t):
    rotation_layer(circ, idx_x[: circ.num_qubits], None, x, t, axis="x", mode="cycle")
    return ent_pairs(circ, None, None, x, t)


def macro_shape_then_data(circ, idx_x, idx_t, x, t):
    nq = circ.num_qubits
    rotation_layer(circ, None, idx_t[:nq], x, t, axis="z", mode="same")
    rotation_layer(circ, idx_x[:nq], None, x, t, axis="x", mode="cycle")
    return circ


def macro_data_then_shape(circ, idx_x, idx_t, x, t):
    nq = circ.num_qubits
    rotation_layer(circ, idx_x[:nq], None, x, t, axis="x", mode="cycle")
    rotation_layer(circ, None, idx_t[:nq], x, t, axis="z", mode="same")
    return circ


def macro_data_axis_mix(circ, idx_x, idx_t, x, t):
    nq = circ.num_qubits
    rotation_layer(circ, idx_x[:nq], None, x, t, axis="x", mode="cycle")
    rotation_layer(circ, idx_x[:nq], None, x, t, axis="z", mode="cycle")
    return circ


def stop(circ, idx_x, idx_t, x, t):
    return circ


def _tag(fn, name, kx, kt):
    fn.name = name
    fn.kx = kx
    fn.kt = kt
    return fn


_tag(ent_none, "ent_none", 0, 0)
_tag(ent_chain, "ent_chain", 0, 0)
_tag(ent_ring, "ent_ring", 0, 0)
_tag(ent_pairs, "ent_pairs", 0, 0)
_tag(basis_all_h, "basis_all_h", 0, 0)
_tag(basis_even_h, "basis_even_h", 0, 0)
_tag(basis_odd_h, "basis_odd_h", 0, 0)
_tag(mix_shape_basis_data, "mix_shape_basis_data", lambda nq: nq, lambda nq: nq)
_tag(macro_data_entangle_chain, "macro_data_entangle_chain", lambda nq: nq, 0)
_tag(macro_data_entangle_pairs, "macro_data_entangle_pairs", lambda nq: nq, 0)
_tag(macro_shape_then_data, "macro_shape_then_data", lambda nq: nq, lambda nq: nq)
_tag(macro_data_then_shape, "macro_data_then_shape", lambda nq: nq, lambda nq: nq)
_tag(macro_data_axis_mix, "macro_data_axis_mix", lambda nq: nq, 0)
_tag(stop, "stop", 0, 0)


def build_patterns(nq: int):
    rotations = [make_rot_pattern(axis, mode, nq) for axis in ("x", "z") for mode in ("same", "cycle")]
    return rotations + [
        ent_none,
        ent_pairs,
        ent_chain,
        ent_ring,
        basis_even_h,
        basis_all_h,
        basis_odd_h,
        mix_shape_basis_data,
        mix_data_entangle(ent_pairs),
        mix_data_entangle(ent_chain),
        macro_data_entangle_pairs,
        macro_data_entangle_chain,
        macro_shape_then_data,
        macro_data_then_shape,
        macro_data_axis_mix,
        stop,
    ]


__all__ = [
    "ROTATIONS",
    "basis_all_h",
    "basis_even_h",
    "basis_odd_h",
    "build_patterns",
    "ent_chain",
    "ent_none",
    "ent_pairs",
    "ent_ring",
    "macro_data_axis_mix",
    "macro_data_entangle_chain",
    "macro_data_entangle_pairs",
    "macro_data_then_shape",
    "macro_shape_then_data",
    "make_rot_pattern",
    "mix_data_entangle",
    "mix_shape_basis_data",
    "rotation_layer",
    "stop",
]
