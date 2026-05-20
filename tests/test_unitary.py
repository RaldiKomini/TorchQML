import pytest
import torch

from TorchQML.core.circuit import _kron_all
from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.unitary import UnitarySimulator
from TorchQML.gates import CNOT, Gate, H, I, S, T, Z


def test_simulator_starts_and_resets_to_identity():
    sim = UnitarySimulator(num_qubits=2)

    assert torch.allclose(sim.unitary, torch.eye(4, dtype=DTYPE, device=DEVICE))
    assert sim.is_unitary()

    sim.add_gate(H, 0)
    sim.add_gate(T, 1)
    sim.reset()

    assert torch.allclose(sim.unitary, torch.eye(4, dtype=DTYPE, device=DEVICE))
    assert sim.gate_layers == []


def test_single_qubit_gate_expansion_follows_wire_order():
    left = UnitarySimulator(num_qubits=2)
    right = UnitarySimulator(num_qubits=2)

    left.add_gate(H, 0)
    right.add_gate(H, 1)

    assert torch.allclose(left.unitary, _kron_all([I.matrix, H.matrix]))
    assert torch.allclose(right.unitary, _kron_all([H.matrix, I.matrix]))


def test_cnot_supports_both_control_target_orders():
    sim = UnitarySimulator(num_qubits=2)

    sim.add_cnot(0, 1)
    assert torch.allclose(sim.unitary, CNOT(2, 0, 1).matrix)
    assert sim.is_unitary()

    sim.reset()
    sim.add_cnot(1, 0)
    assert torch.allclose(sim.unitary, CNOT(2, 1, 0).matrix)
    assert sim.is_unitary()


def test_gates_are_left_multiplied():
    sim = UnitarySimulator(num_qubits=1)

    sim.add_gate(H, 0)
    sim.add_gate(Z, 0)

    assert torch.allclose(sim.unitary, Z.matrix @ H.matrix)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        UnitarySimulator(num_qubits=0)

    sim = UnitarySimulator(num_qubits=2)

    with pytest.raises(ValueError):
        sim.add_gate(S, 2)

    with pytest.raises(ValueError):
        sim.add_cnot(0, 0)

    with pytest.raises(ValueError):
        sim.add_full(Gate(torch.eye(2, dtype=DTYPE, device=DEVICE), "too-small"), (0,))


def test_counts_are_updated_as_gates_are_added():
    sim = UnitarySimulator(2)

    sim.add_gate(T, 0)
    sim.add_gate(H, 1)
    sim.add_cnot(0, 1)

    assert sim.counts() == {"depth": 3, "t_count": 1, "cnot_count": 1}


def test_counts_are_updated_as_gates_are_added2():
    sim = UnitarySimulator(2)

    sim.add_gate(T, 0)
    sim.add_gate(H, 1)
    sim.add_cnot(0, 1)
    sim.add_cnot(1, 0)
    sim.add_cnot(1,0)
    sim.add_gate(T, 1)
    sim.add_gate(H, 1)

    assert sim.counts() == {"depth": 7, "t_count": 2, "cnot_count": 3}
