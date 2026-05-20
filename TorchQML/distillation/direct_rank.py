import math

from TorchQML.core.circuit import Circuit, CircuitSpec
from TorchQML.core.sym import Tsym, Xsym
from TorchQML.gates import H, I
from TorchQML.gates.vgate import ham1q, ham2q_full


DEFAULT_SLOT_SPEC = (
    ("1q", 0),
    ("1q", 1),
    ("1q", 2),
    ("1q", 3),
    ("2q", (0, 1)),
    ("2q", (1, 2)),
    ("2q", (2, 3)),
    ("2q", (0, 3)),
    ("2q", (0, 2)),
    ("2q", (1, 3)),
    ("1q", 3),
    ("1q", 2),
    ("1q", 1),
    ("1q", 0),
    ("2q", (0, 2)),
    ("2q", (1, 3)),
    ("2q", (0, 1)),
    ("2q", (2, 3)),
    ("2q", (1, 2)),
    ("2q", (0, 3)),
    ("1q", 0),
    ("1q", 2),
    ("1q", 1),
    ("1q", 3),
    ("2q", (0, 1)),
    ("2q", (1, 2)),
    ("2q", (2, 3)),
    ("2q", (0, 3)),
    ("2q", (0, 2)),
    ("2q", (1, 3)),
    ("1q", 0),
    ("1q", 1),
    ("1q", 2),
    ("1q", 3),
)


def direct_rank_tlen(slot_spec=DEFAULT_SLOT_SPEC) -> int:
    return 5 * len(slot_spec)


def build_direct_rank_hamiltonian_circuit(
    spec: CircuitSpec,
    slot_spec=DEFAULT_SLOT_SPEC,
    *,
    add_initial_h: bool = True,
) -> Circuit:
    x = Xsym()
    t = Tsym()
    circ = Circuit(num_qubits=spec.num_qubits, specs=spec)
    if add_initial_h:
        circ.add_gates([H] * spec.num_qubits)

    t_idx = 0
    for slot_idx, (kind, target) in enumerate(slot_spec):
        i1 = slot_idx % spec.xlen
        lam = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
        t_idx += 2
        wx = t[t_idx]
        wy = t[t_idx + 1]
        wz = t[t_idx + 2]
        t_idx += 3

        if kind == "1q":
            layer = [I] * spec.num_qubits
            layer[target] = ham1q(lam * wx, lam * wy, lam * wz)
            circ.add_gates(layer)
        elif kind == "2q":
            q1, q2 = target
            circ.add_full(ham2q_full(spec.num_qubits, q1, q2, lam * wx, lam * wy, lam * wz))
        else:
            raise ValueError(f"unknown slot kind: {kind}")

    return circ


state_ham_dir = build_direct_rank_hamiltonian_circuit


__all__ = [
    "DEFAULT_SLOT_SPEC",
    "build_direct_rank_hamiltonian_circuit",
    "direct_rank_tlen",
    "state_ham_dir",
]
