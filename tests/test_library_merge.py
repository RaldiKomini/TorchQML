import torch

import TorchQML
from TorchQML.core.unitary import UnitarySimulator
from TorchQML.encoding import AmpEnc
from TorchQML.gates import CNOT
from TorchQML.kernels import kernel_matrix
from TorchQML.qas import build_patterns
from TorchQML.states.states2q_hardware import rh2q
from TorchQML.states.states4q import state4_small
from TorchQML.states.states4q_basic import state4_small as split_state4_small
from TorchQML.states.states4q_entanglement import state4_entangle_improved


def test_top_level_imports_expose_merged_modules():
    assert hasattr(TorchQML, "kernels")
    assert hasattr(TorchQML, "qas")
    assert hasattr(TorchQML, "synthesis")
    assert hasattr(TorchQML, "states")
    assert hasattr(TorchQML, "quantum_svm")


def test_unitary_simulator_cnot_matches_gate_cnot_for_all_orders():
    for control, target in [(0, 1), (1, 0), (0, 2), (2, 1)]:
        sim = UnitarySimulator(3)
        sim.add_cnot(control, target)
        assert torch.allclose(sim.unitary, CNOT(3, control, target).matrix)


def test_amplitude_encoder_batches_pad_and_normalize():
    enc = AmpEnc(3)
    x = torch.ones(2, 5)
    psi = enc(x)
    assert psi.shape == (2, 8)
    assert torch.allclose(torch.linalg.vector_norm(psi, dim=1), torch.ones(2))


def test_kernel_matrix_accepts_amplitude_encoding():
    circ = TorchQML.Circuit(num_qubits=2, specs=TorchQML.CircuitSpec(2, 3, 1))
    theta = torch.zeros(1, dtype=torch.complex64)
    X = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    K = kernel_matrix(X, circ, theta, amp_enc=AmpEnc(2))
    assert K.shape == (2, 2)
    assert torch.allclose(K, K.T)


def test_split_state_wrapper_preserves_old_import():
    assert state4_small is split_state4_small


def test_qas_pattern_registry_is_available():
    names = {getattr(pattern, "name", "") for pattern in build_patterns(2)}
    assert {"ent_chain", "ent_ring", "stop"}.issubset(names)


def test_repaired_state_templates_build_circuits():
    spec = TorchQML.CircuitSpec(num_qubits=4, xlen=16, tlen=96)
    assert state4_entangle_improved(spec).num_qubits == 4
    assert rh2q(20, depth=1, reuploads_per_block=1).num_qubits == 2
