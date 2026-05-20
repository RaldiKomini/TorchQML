import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.unitary import UnitarySimulator
from TorchQML.gates import H, T, Tdg


def target_t() -> torch.Tensor:
    """Single-qubit T target."""
    sim = UnitarySimulator(num_qubits=1)
    sim.add_gate(T, 0)
    return sim.unitary.clone()


def target_bell() -> torch.Tensor:
    """Bell-preparation unitary target."""
    sim = UnitarySimulator(num_qubits=2)
    sim.add_gate(H, 0)
    sim.add_cnot(0, 1)
    return sim.unitary.clone()


def target_ghz() -> torch.Tensor:
    """Three-qubit GHZ-preparation unitary target."""
    sim = UnitarySimulator(num_qubits=3)
    sim.add_gate(H, 0)
    sim.add_cnot(0, 1)
    sim.add_cnot(1, 2)
    return sim.unitary.clone()


def target_toffoli() -> torch.Tensor:
    """Direct CCNOT matrix with controls q0/q1 and target q2."""
    dim = 8
    matrix = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)

    for basis in range(dim):
        q0 = (basis >> 0) & 1
        q1 = (basis >> 1) & 1
        output = basis
        if q0 and q1:
            output = basis ^ (1 << 2)
        matrix[output, basis] = 1.0

    return matrix


def target_toffoli_prefix(depth: int) -> torch.Tensor:
    """Target unitary for the first `depth` gates of the Toffoli baseline."""
    gates = [
        ("H", 2),
        ("CNOT", 1, 2),
        ("Tdg", 2),
        ("CNOT", 0, 2),
        ("T", 2),
        ("CNOT", 1, 2),
        ("Tdg", 2),
        ("CNOT", 0, 2),
        ("T", 1),
        ("T", 2),
        ("CNOT", 0, 1),
        ("H", 2),
        ("T", 0),
        ("Tdg", 1),
        ("CNOT", 0, 1),
    ]

    if not 0 <= depth <= len(gates):
        raise ValueError("Toffoli prefix depth must be between 0 and 15")

    sim = UnitarySimulator(num_qubits=3)

    for gate in gates[:depth]:
        if gate[0] == "H":
            sim.add_gate(H, gate[1])
        elif gate[0] == "T":
            sim.add_gate(T, gate[1])
        elif gate[0] == "Tdg":
            sim.add_gate(Tdg, gate[1])
        elif gate[0] == "CNOT":
            sim.add_cnot(gate[1], gate[2])

    return sim.unitary.clone()


TARGETS = {
    "t": target_t,
    "bell": target_bell,
    "ghz": target_ghz,
    "toffoli": target_toffoli,
}
