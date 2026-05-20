from TorchQML.synthesis.baselines import build_baseline_unitary
from TorchQML.synthesis.metrics import unitary_fidelity
from TorchQML.synthesis.targets import TARGETS, target_toffoli, target_toffoli_prefix


def test_baseline_unitaries_match_targets():
    for name, target_fn in TARGETS.items():
        fidelity = unitary_fidelity(build_baseline_unitary(name), target_fn())
        assert fidelity > 0.999


def test_toffoli_prefix_15_matches_toffoli_target():
    fidelity = unitary_fidelity(target_toffoli_prefix(15), target_toffoli())

    assert fidelity > 0.999
