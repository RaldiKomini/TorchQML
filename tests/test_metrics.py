import pytest
import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.synthesis.metrics import simplified_depth, unitary_distance, unitary_fidelity


def test_identical_unitaries_have_fidelity_one():
    unitary = torch.eye(2, dtype=DTYPE, device=DEVICE)

    assert unitary_fidelity(unitary, unitary) == pytest.approx(1.0)
    assert unitary_distance(unitary, unitary) == pytest.approx(0.0)


def test_global_phase_does_not_change_fidelity():
    unitary = torch.eye(2, dtype=DTYPE, device=DEVICE)
    phased = 1j * unitary

    assert unitary_fidelity(unitary, phased) == pytest.approx(1.0)
    assert unitary_distance(unitary, phased) == pytest.approx(0.0)


def test_shape_mismatch_is_rejected():
    one_qubit = torch.eye(2, dtype=DTYPE, device=DEVICE)
    two_qubit = torch.eye(4, dtype=DTYPE, device=DEVICE)

    with pytest.raises(ValueError):
        unitary_fidelity(one_qubit, two_qubit)


def test_simplified_depth_removes_adjacent_cancellations():
    circuit = [
        {"name": "H(0)", "qubits": [0]},
        {"name": "H(0)", "qubits": [0]},
        {"name": "T(0)", "qubits": [0]},
        {"name": "Tdg(0)", "qubits": [0]},
        {"name": "CNOT(c=0,t=1)", "qubits": [0, 1]},
        {"name": "CNOT(c=0,t=1)", "qubits": [0, 1]},
        {"name": "S(1)", "qubits": [1]},
    ]

    assert simplified_depth(circuit) == 1
