import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from TorchQML.core.unitary import UnitarySimulator
from TorchQML.synthesis.actions import (
    CircuitAction,
    actions_cancel,
    apply_action,
    build_actions,
    primitive_actions,
)
from TorchQML.synthesis.metrics import simplified_depth, unitary_distance, unitary_fidelity


class CircuitSynthesisDistillEnv(gym.Env):
    """Gymnasium environment for building a target unitary gate by gate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        target,
        probe_states,
        num_qubits: int,
        max_depth: int = 20,
        success_threshold: float = 0.999,
        depth_penalty: float = 0.01,
        t_penalty: float = 0.02,
        cnot_penalty: float = 0.0,
        actions=None,
        block_inverse_actions: bool = False,
        max_steps: int | None = None,
        invalid_action_penalty: float = 0.05,
        start_unitary=None,
    ):
        """Create a synthesis task for one fixed target unitary."""
        super().__init__()

        self.target = target
        self.probe_states = probe_states.to(dtype=target.dtype, device=target.device)
        self.teacher_outputs = self.probe_states @ self.target.T
        identity_outputs = self.probe_states
        displacement = self.teacher_outputs - identity_outputs

        weights = torch.linalg.vector_norm(displacement, dim=1).square()
        weights = weights / weights.sum().clamp_min(1e-12)

        self.probe_weights = weights
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.success_threshold = success_threshold
        self.depth_penalty = depth_penalty
        self.t_penalty = t_penalty
        self.cnot_penalty = cnot_penalty
        self.block_inverse_actions = block_inverse_actions
        self.max_steps = max_steps if max_steps is not None else max_depth
        self.invalid_action_penalty = invalid_action_penalty
        self.start_unitary = start_unitary

        dim = 1 << num_qubits
        if self.start_unitary is not None and self.start_unitary.shape != (dim, dim):
            raise ValueError("start_unitary has incompatible shape")

        self.simulator = UnitarySimulator(num_qubits)
        self.actions = actions if actions is not None else build_actions(num_qubits)
        self.steps = 0
        self._reset_simulator()
        self.previous_fidelity = self._state_fidelity()
        self.best_fidelity = self.previous_fidelity

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, dim, dim),
            dtype=np.float32,
        )

    def _state_fidelity(self) -> float:
        """Weighted state fidelity between teacher and current student."""
        student_outputs = self.probe_states @ self.simulator.unitary.T
        overlaps = (self.teacher_outputs.conj() * student_outputs).sum(dim=1)
        fidelities = overlaps.abs().square().real

        weighted_fidelity = (self.probe_weights * fidelities).sum()
        return float(weighted_fidelity)


    def _get_obs(self):
        """Return real and imaginary parts of the current unitary."""
        unitary = self.simulator.unitary.detach().cpu().numpy()
        return np.stack([unitary.real, unitary.imag]).astype(np.float32)

    def _get_circuit(self):
        """Return the primitive gate log in JSON-friendly form."""
        return [
            {"name": gate.name, "qubits": list(gate.qubits)}
            for gate in self.simulator.gate_layers
        ]

    def _reset_simulator(self):
        """Reset to identity or to the provided curriculum start unitary."""
        self.simulator.reset()

        if self.start_unitary is not None:
            # The start unitary is not counted as part of the discovered circuit.
            self.simulator.unitary = self.start_unitary.to(
                dtype=self.simulator.unitary.dtype,
                device=self.simulator.unitary.device,
            ).clone()

    def _last_action(self):
        """Return the last primitive action, if any."""
        if not self.simulator.gate_layers:
            return None

        gate = self.simulator.gate_layers[-1]
        return CircuitAction(gate.name, gate.qubits)

    def _would_cancel(self, parts):
        """Check whether an action would immediately undo recent work."""
        last_action = self._last_action()

        if last_action is not None and actions_cancel(last_action, parts[0]):
            return True

        for left, right in zip(parts, parts[1:]):
            if actions_cancel(left, right):
                return True

        return False

    def reset(self, seed=None, options=None):
        """Start a new episode."""
        super().reset(seed=seed)

        self._reset_simulator()
        self.steps = 0

        self.previous_fidelity = self._state_fidelity()
        self.best_fidelity = self.previous_fidelity

        unitary_fid = unitary_fidelity(self.simulator.unitary, self.target)

        info = {
            "fidelity": self.previous_fidelity,          # main training fidelity = state fidelity
            "state_fidelity": self.previous_fidelity,
            "unitary_fidelity": unitary_fid,
            "best_fidelity": self.best_fidelity,
            "distance": 1.0 - self.previous_fidelity,
            "unitary_distance": 1.0 - unitary_fid,
            "circuit": self._get_circuit(),
            "simplified_depth": simplified_depth(self._get_circuit()),
            **self.simulator.counts(),
        }

        return self._get_obs(), info

    def step(self, action_idx):
        """Apply one action and return the Gymnasium step tuple."""
        action = self.actions[action_idx]
        parts = primitive_actions(action)
        self.steps += 1

        old_fidelity = self.previous_fidelity
        gate_count = len(parts)
        t_gate_count = sum(1 for part in parts if part.name in {"T", "Tdg"})
        cnot_gate_count = sum(1 for part in parts if part.name == "CNOT")
        blocked = self.block_inverse_actions and self._would_cancel(parts)
        too_deep = self.simulator.depth + gate_count > self.max_depth

        if blocked or too_deep:
            new_fidelity = old_fidelity
            reward = -self.invalid_action_penalty
        else:
            apply_action(self.simulator, action)

            new_fidelity = self._state_fidelity()

            reward = 10.0 * (new_fidelity - old_fidelity)
            reward -= self.depth_penalty * gate_count
            reward -= self.t_penalty * t_gate_count
            reward -= self.cnot_penalty * cnot_gate_count

        self.best_fidelity = max(self.best_fidelity, new_fidelity)

        distance = 1.0 - new_fidelity

        terminated = new_fidelity >= self.success_threshold
        truncated = (
            self.simulator.depth >= self.max_depth
            or self.steps >= self.max_steps
            or too_deep
        )

        if terminated:
            reward += 5.0

        self.previous_fidelity = new_fidelity

        unitary_fid = unitary_fidelity(self.simulator.unitary, self.target)

        info = {
            "fidelity": new_fidelity,
            "state_fidelity": new_fidelity,
            "unitary_fidelity": unitary_fid,
            "best_fidelity": self.best_fidelity,
            "distance": 1.0 - new_fidelity,
            "unitary_distance": 1.0 - unitary_fid,
            "action": action.name,
            "qubits": action.qubits,
            "blocked_action": blocked,
            "too_deep_action": too_deep,
            "circuit": self._get_circuit(),
            "simplified_depth": simplified_depth(self._get_circuit()),
            **self.simulator.counts(),
        }

        return self._get_obs(), reward, terminated, truncated, info
