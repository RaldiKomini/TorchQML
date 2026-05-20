import numpy as np

from TorchQML.gates import T, Tdg
from TorchQML.synthesis.actions import CircuitAction
from TorchQML.synthesis.env import CircuitSynthesisEnv
from TorchQML.synthesis.targets import target_t


def test_env_reset_returns_valid_observation_and_info():
    env = CircuitSynthesisEnv(target=target_t(), num_qubits=1)

    obs, info = env.reset()

    assert obs.shape == (2, 2, 2)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "fidelity" in info
    assert "best_fidelity" in info
    assert "distance" in info
    assert info["circuit"] == []
    assert info["depth"] == 0
    assert info["simplified_depth"] == 0


def test_env_step_updates_state_and_returns_gymnasium_tuple():
    env = CircuitSynthesisEnv(target=target_t(), num_qubits=1, max_depth=5)

    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (2, 2, 2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["depth"] == 1
    assert len(info["circuit"]) == 1
    assert info["simplified_depth"] == 1


def test_env_truncates_at_max_depth():
    env = CircuitSynthesisEnv(target=target_t(), num_qubits=1, max_depth=1)

    env.reset()
    _, _, _, truncated, info = env.step(0)

    assert truncated
    assert info["depth"] == 1


def test_env_can_apply_macro_action():
    macro = CircuitAction(
        "T(0)+T(0)",
        (),
        parts=(
            CircuitAction("T", (0,), T),
            CircuitAction("T", (0,), T),
        ),
    )
    env = CircuitSynthesisEnv(
        target=target_t(),
        num_qubits=1,
        max_depth=2,
        actions=[macro],
    )

    env.reset()
    _, _, _, truncated, info = env.step(0)

    assert truncated
    assert info["depth"] == 2
    assert info["t_count"] == 2


def test_env_can_block_immediate_inverse_action():
    actions = [
        CircuitAction("T", (0,), T),
        CircuitAction("Tdg", (0,), Tdg),
    ]
    env = CircuitSynthesisEnv(
        target=target_t(),
        num_qubits=1,
        max_depth=3,
        actions=actions,
        block_inverse_actions=True,
    )

    env.reset()
    env.step(0)
    _, _, _, _, info = env.step(1)

    assert info["blocked_action"]
    assert info["depth"] == 1
    assert info["simplified_depth"] == 1


def test_env_can_start_from_custom_unitary():
    start = target_t()
    env = CircuitSynthesisEnv(
        target=target_t(),
        start_unitary=start,
        num_qubits=1,
        max_depth=1,
    )

    _, info = env.reset()

    assert info["fidelity"] > 0.999
    assert info["depth"] == 0
    assert info["circuit"] == []
