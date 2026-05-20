import numpy as np
import torch

from TorchQML.gates import H, S, T, Tdg
from TorchQML.synthesis.actions import CircuitAction
from TorchQML.synthesis.env_all import CircuitSynthesisAllEnv


def make_probe_states(target):
    return torch.eye(target.shape[0], dtype=target.dtype, device=target.device)


def test_all_env_masks_immediate_inverse_action():
    actions = [
        CircuitAction("T", (0,), T),
        CircuitAction("Tdg", (0,), Tdg),
    ]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=3,
        actions=actions,
        block_inverse_actions=True,
    )

    env.reset()
    env.step(0)

    mask = env.valid_action_mask()

    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False]
    assert env.action_masks().tolist() == [True, False]


def test_all_env_truncates_blocked_action_by_default():
    actions = [
        CircuitAction("T", (0,), T),
        CircuitAction("Tdg", (0,), Tdg),
    ]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=3,
        actions=actions,
        block_inverse_actions=True,
    )

    env.reset()
    env.step(0)
    _, _, terminated, truncated, info = env.step(1)

    assert not terminated
    assert truncated
    assert info["invalid_action"]
    assert info["blocked_action"]
    assert info["invalid_action_count"] == 1
    assert info["depth"] == 1
    assert info["action_mask"] == [True, False]


def test_all_env_masks_unitary_cycle():
    actions = [CircuitAction("S", (0,), S)]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=8,
        actions=actions,
        block_inverse_actions=False,
        cycle_pruning=True,
    )

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)

    assert env.valid_action_mask().tolist() == [False]

    _, _, _, truncated, info = env.step(0)

    assert truncated
    assert info["invalid_action"]
    assert info["cycle_action"]
    assert info["depth"] == 3


def test_all_env_masks_approximate_t_power_cycle():
    actions = [CircuitAction("T", (0,), T)]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=10,
        actions=actions,
        block_inverse_actions=False,
        cycle_pruning=True,
    )

    env.reset()
    for _ in range(7):
        env.step(0)

    assert env.valid_action_mask().tolist() == [False]

    _, _, _, truncated, info = env.step(0)

    assert truncated
    assert info["invalid_action"]
    assert info["cycle_action"]
    assert info["depth"] == 7


def test_all_env_tracks_best_unitary_fidelity():
    actions = [CircuitAction("T", (0,), T)]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=1,
        actions=actions,
    )

    _, reset_info = env.reset()
    _, _, _, _, step_info = env.step(0)

    assert step_info["best_unitary_fidelity"] >= reset_info["unitary_fidelity"]


def test_all_env_can_start_from_reverse_neighborhood():
    actions = [CircuitAction("T", (0,), T)]
    target = H.matrix.clone()
    env = CircuitSynthesisAllEnv(
        target=target,
        probe_states=make_probe_states(target),
        num_qubits=1,
        max_depth=3,
        actions=actions,
        reverse_neighborhood_probability=1.0,
        reverse_neighborhood_depth=1,
    )

    _, info = env.reset(seed=0)

    assert info["reverse_neighborhood_active"]
    assert info["reverse_neighborhood_depth"] == 1
    assert info["depth"] == 0
    assert info["circuit"] == []
