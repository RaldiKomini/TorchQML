from TorchQML.synthesis.search_mitm import apply_action_pruning
from TorchQML.synthesis.train_all import make_env


def h_qubits(env):
    return [action.qubits[0] for action in env.actions if action.name == "H"]


def test_monomial_h_pruning_removes_h_from_phase_target():
    env = make_env("t", action_set="minimal")

    metadata, original_indices = apply_action_pruning(env, "monomial-h", 1e-5)

    assert metadata["monomial_target"] is True
    assert metadata["selected_h_qubits"] == []
    assert metadata["original_action_count"] == 3
    assert [action.name for action in env.actions] == ["T", "Tdg"]
    assert original_indices == [1, 2]


def test_monomial_h_pruning_keeps_toffoli_target_h_only():
    env = make_env("toffoli", action_set="minimal")

    metadata, original_indices = apply_action_pruning(env, "monomial-h", 1e-5)

    assert metadata["monomial_target"] is True
    assert metadata["selected_h_qubits"] == [2]
    assert metadata["original_action_count"] == 15
    assert len(env.actions) == 13
    assert h_qubits(env) == [2]
    assert original_indices == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_monomial_h_pruning_leaves_non_monomial_target_unpruned():
    env = make_env("bell", action_set="minimal")
    original_count = len(env.actions)

    metadata, original_indices = apply_action_pruning(env, "monomial-h", 1e-5)

    assert metadata["monomial_target"] is False
    assert len(env.actions) == original_count
    assert original_indices == list(range(original_count))
