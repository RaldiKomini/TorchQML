import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from TorchQML.synthesis.actions_toffoli import build_toffoli_actions
from TorchQML.core.unitary import UnitarySimulator
from TorchQML.synthesis.actions import (
    CircuitAction,
    actions_cancel,
    apply_action,
    build_actions,
    primitive_actions,
)
from TorchQML.synthesis.metrics import simplified_depth, unitary_fidelity


class CircuitSynthesisAllEnv(gym.Env):
    """Behavior-aware circuit synthesis environment.

    Reward is based on a combined score:
      basis action score
    + weighted state response score
    + observable transformation score

    Exact unitary fidelity is still reported for evaluation.
    """

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
        cnot_penalty: float = 0.02,
        actions=None,
        block_inverse_actions: bool = True,
        max_steps: int | None = None,
        invalid_action_penalty: float = 0.05,
        invalid_action_patience: int = 1,
        cycle_pruning: bool = True,
        cycle_key_decimals: int = 5,
        stagnation_penalty: float = 0.02,
        start_unitary=None,
        reverse_neighborhood_probability: float = 0.0,
        reverse_neighborhood_depth: int = 0,
        basis_weight: float = 0.25,
        state_weight: float = 0.50,
        observable_weight: float = 0.25,
        uniform_probe_mix: float = 0.20,
    ):
        super().__init__()

        self.target = target
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.success_threshold = success_threshold
        self.depth_penalty = depth_penalty
        self.t_penalty = t_penalty
        self.cnot_penalty = cnot_penalty
        self.block_inverse_actions = block_inverse_actions
        self.max_steps = max_steps if max_steps is not None else max_depth
        self.invalid_action_penalty = invalid_action_penalty
        if invalid_action_patience < 1:
            raise ValueError("invalid_action_patience must be at least 1.")
        self.invalid_action_patience = invalid_action_patience
        self.cycle_pruning = cycle_pruning
        self.cycle_key_decimals = cycle_key_decimals
        self.start_unitary = start_unitary
        self.reverse_neighborhood_probability = float(reverse_neighborhood_probability)
        self.reverse_neighborhood_probability = max(
            0.0,
            min(1.0, self.reverse_neighborhood_probability),
        )
        if reverse_neighborhood_depth < 0:
            raise ValueError("reverse_neighborhood_depth must be non-negative.")
        self.reverse_neighborhood_depth = reverse_neighborhood_depth

        self.basis_weight = basis_weight
        self.state_weight = state_weight
        self.observable_weight = observable_weight
        self.stagnation_penalty = stagnation_penalty

        weight_sum = basis_weight + state_weight + observable_weight
        if weight_sum <= 0:
            raise ValueError("At least one behavior score weight must be positive.")

        self.basis_weight /= weight_sum
        self.state_weight /= weight_sum
        self.observable_weight /= weight_sum

        dim = 1 << num_qubits
        self.dim = dim

        if self.target.shape != (dim, dim):
            raise ValueError("target has incompatible shape")

        if self.start_unitary is not None and self.start_unitary.shape != (dim, dim):
            raise ValueError("start_unitary has incompatible shape")

        self.probe_states = probe_states.to(dtype=target.dtype, device=target.device)
        self.teacher_outputs = self.probe_states @ self.target.T

        self.probe_weights = self._make_probe_weights(uniform_probe_mix)

        self.basis_states = torch.eye(
            dim,
            dtype=target.dtype,
            device=target.device,
        )
        self.teacher_basis_outputs = self.basis_states @ self.target.T

        self.observables = self._make_observables()
        self.teacher_observable_transforms = [
            self._transform_observable(self.target, obs)
            for obs in self.observables
        ]

        self.simulator = UnitarySimulator(num_qubits)
        self.actions = actions if actions is not None else build_actions(num_qubits)
        self.action_unitaries = self._make_action_unitaries()

        self.steps = 0
        self.invalid_action_count = 0
        self.visited_unitary_keys = set()
        self.reverse_neighborhood_active = False
        self.reverse_neighborhood_depth_used = 0
        self._reset_simulator()

        identity_scores = self._raw_scores()
        self.identity_behavior_score = identity_scores["behavior_score"]

        initial_scores = self._scores()
        self.previous_fidelity = initial_scores["behavior_score"]
        self.best_fidelity = self.previous_fidelity
        self.best_unitary_fidelity = unitary_fidelity(self.simulator.unitary, self.target)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, dim, dim),
            dtype=np.float32,
        )

    def _make_probe_weights(self, uniform_probe_mix: float) -> torch.Tensor:
        """Give more weight to probes where target differs from identity."""
        identity_outputs = self.probe_states
        displacement = self.teacher_outputs - identity_outputs

        weights = torch.linalg.vector_norm(displacement, dim=1).square()
        weights = weights / weights.sum().clamp_min(1e-12)

        uniform = torch.full_like(weights, 1.0 / weights.numel())

        uniform_probe_mix = float(uniform_probe_mix)
        uniform_probe_mix = max(0.0, min(1.0, uniform_probe_mix))

        return (1.0 - uniform_probe_mix) * weights + uniform_probe_mix * uniform

    def _kron_all(self, mats: list[torch.Tensor]) -> torch.Tensor:
        out = mats[0]
        for mat in mats[1:]:
            out = torch.kron(out, mat)
        return out

    def _single_qubit_pauli(self, name: str, qubit: int) -> torch.Tensor:
        dtype = self.target.dtype
        device = self.target.device

        eye = torch.eye(2, dtype=dtype, device=device)

        if name == "X":
            pauli = torch.tensor(
                [[0, 1], [1, 0]],
                dtype=dtype,
                device=device,
            )
        elif name == "Y":
            pauli = torch.tensor(
                [[0, -1j], [1j, 0]],
                dtype=dtype,
                device=device,
            )
        elif name == "Z":
            pauli = torch.tensor(
                [[1, 0], [0, -1]],
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError(f"Unknown Pauli {name}")

        mats = [pauli if q == qubit else eye for q in range(self.num_qubits)]
        return self._kron_all(mats)

    def _make_observables(self) -> list[torch.Tensor]:
        """Local Pauli observables X_i, Y_i, Z_i."""
        observables = []
        for q in range(self.num_qubits):
            observables.append(self._single_qubit_pauli("X", q))
            observables.append(self._single_qubit_pauli("Y", q))
            observables.append(self._single_qubit_pauli("Z", q))
        return observables

    def _transform_observable(self, unitary: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
        """Heisenberg-picture observable transform: U^dagger O U."""
        return unitary.conj().T @ observable @ unitary

    def _state_score(self) -> float:
        """Weighted output-state agreement on probe states."""
        student_outputs = self.probe_states @ self.simulator.unitary.T
        overlaps = (self.teacher_outputs.conj() * student_outputs).sum(dim=1)
        fidelities = overlaps.abs().square().real
        return float((self.probe_weights * fidelities).sum().item())

    def _basis_score(self) -> float:
        """Output agreement on all computational basis states."""
        student_outputs = self.basis_states @ self.simulator.unitary.T
        overlaps = (self.teacher_basis_outputs.conj() * student_outputs).sum(dim=1)
        fidelities = overlaps.abs().square().real
        return float(fidelities.mean().item())

    def _observable_score(self) -> float:
        """Agreement of local Pauli observable transformations."""
        current = self.simulator.unitary
        scores = []

        for target_transformed, obs in zip(self.teacher_observable_transforms, self.observables):
            current_transformed = self._transform_observable(current, obs)

            overlap = torch.trace(target_transformed.conj().T @ current_transformed)
            score = overlap.abs().square().real / float(self.dim * self.dim)
            scores.append(score)

        return float(torch.stack(scores).mean().item())

    def _raw_scores(self) -> dict[str, float]:
        basis = self._basis_score()
        state = self._state_score()
        observable = self._observable_score()

        behavior = (
            self.basis_weight * basis
            + self.state_weight * state
            + self.observable_weight * observable
        )

        return {
            "behavior_score": float(behavior),
            "basis_score": basis,
            "state_score": state,
            "observable_score": observable,
        }

    def _scores(self) -> dict[str, float]:
        raw = self._raw_scores()

        denom = 1.0 - self.identity_behavior_score
        if denom <= 1e-12:
            normalized_behavior = raw["behavior_score"]
        else:
            normalized_behavior = (raw["behavior_score"] - self.identity_behavior_score) / denom

        normalized_behavior = min(1.0, normalized_behavior)

        return {
            **raw,
            "raw_behavior_score": raw["behavior_score"],
            "behavior_score": normalized_behavior,
        }

    def _get_obs(self):
        unitary = self.simulator.unitary.detach().cpu().numpy()
        return np.stack([unitary.real, unitary.imag]).astype(np.float32)

    def _get_circuit(self):
        return [
            {"name": gate.name, "qubits": list(gate.qubits)}
            for gate in self.simulator.gate_layers
        ]

    def _make_action_unitaries(self) -> list[torch.Tensor]:
        matrices = []

        for action in self.actions:
            simulator = UnitarySimulator(self.num_qubits)
            apply_action(simulator, action)
            matrices.append(
                simulator.unitary.to(
                    dtype=self.target.dtype,
                    device=self.target.device,
                )
            )

        return matrices

    def _unitary_key(self, unitary: torch.Tensor) -> bytes:
        canonical = unitary.detach().cpu()
        flat = canonical.flatten()
        nonzero = flat.abs() > 1e-8

        if bool(nonzero.any()):
            pivot = flat[nonzero][0]
            canonical = canonical / (pivot / pivot.abs())

        rounded = torch.round(
            torch.view_as_real(canonical),
            decimals=self.cycle_key_decimals,
        )
        return rounded.numpy().tobytes()

    def _candidate_unitary(self, action_idx: int) -> torch.Tensor:
        return self.action_unitaries[action_idx] @ self.simulator.unitary

    def _sample_reverse_neighborhood_start(self):
        if (
            self.reverse_neighborhood_probability <= 0.0
            or self.reverse_neighborhood_depth <= 0
            or self.np_random.random() >= self.reverse_neighborhood_probability
        ):
            self.reverse_neighborhood_active = False
            self.reverse_neighborhood_depth_used = 0
            return

        unitary = self.target.clone()
        depth = int(self.np_random.integers(1, self.reverse_neighborhood_depth + 1))

        target_key = self._unitary_key(self.target)
        for _ in range(depth):
            action_idx = int(self.np_random.integers(0, len(self.action_unitaries)))
            unitary = self.action_unitaries[action_idx] @ unitary

        if self._unitary_key(unitary) == target_key:
            self.reverse_neighborhood_active = False
            self.reverse_neighborhood_depth_used = 0
            return

        self.simulator.unitary = unitary.clone()
        self.reverse_neighborhood_active = True
        self.reverse_neighborhood_depth_used = depth

    def _reset_simulator(self, allow_reverse_neighborhood: bool = False):
        self.simulator.reset()
        self.reverse_neighborhood_active = False
        self.reverse_neighborhood_depth_used = 0

        if self.start_unitary is not None:
            self.simulator.unitary = self.start_unitary.to(
                dtype=self.simulator.unitary.dtype,
                device=self.simulator.unitary.device,
            ).clone()

        if allow_reverse_neighborhood:
            self._sample_reverse_neighborhood_start()

    def _last_action(self):
        if not self.simulator.gate_layers:
            return None

        gate = self.simulator.gate_layers[-1]
        return CircuitAction(gate.name, gate.qubits)

    def _would_cancel(self, parts):
        last_action = self._last_action()

        if last_action is not None and actions_cancel(last_action, parts[0]):
            return True

        for left, right in zip(parts, parts[1:]):
            if actions_cancel(left, right):
                return True

        return False

    def valid_action_mask(self) -> np.ndarray:
        """Return True for actions that can be applied in the current state."""
        mask = []

        for action_idx, action in enumerate(self.actions):
            parts = primitive_actions(action)
            blocked = self.block_inverse_actions and self._would_cancel(parts)
            too_deep = self.simulator.depth + len(parts) > self.max_depth
            cycle = False

            if self.cycle_pruning and not blocked and not too_deep:
                cycle = (
                    self._unitary_key(self._candidate_unitary(action_idx))
                    in self.visited_unitary_keys
                )

            mask.append(not (blocked or too_deep or cycle))

        return np.asarray(mask, dtype=bool)

    def action_masks(self) -> np.ndarray:
        """Compatibility hook for mask-aware RL algorithms."""
        return self.valid_action_mask()

    def _action_mask_info(self) -> dict[str, object]:
        mask = self.valid_action_mask()
        return {
            "action_mask": mask.tolist(),
            "valid_action_count": int(mask.sum()),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_simulator(allow_reverse_neighborhood=True)
        self.steps = 0
        self.invalid_action_count = 0
        self.visited_unitary_keys = {self._unitary_key(self.simulator.unitary)}

        scores = self._scores()
        self.previous_fidelity = scores["behavior_score"]
        self.best_fidelity = self.previous_fidelity

        unitary_fid = unitary_fidelity(self.simulator.unitary, self.target)
        self.best_unitary_fidelity = unitary_fid

        info = {
            "fidelity": scores["behavior_score"],
            "behavior_score": scores["behavior_score"],
            "basis_score": scores["basis_score"],
            "state_score": scores["state_score"],
            "observable_score": scores["observable_score"],
            "unitary_fidelity": unitary_fid,
            "best_fidelity": self.best_fidelity,
            "best_unitary_fidelity": self.best_unitary_fidelity,
            "distance": 1.0 - scores["behavior_score"],
            "unitary_distance": 1.0 - unitary_fid,
            "circuit": self._get_circuit(),
            "simplified_depth": simplified_depth(self._get_circuit()),
            "raw_behavior_score": scores["raw_behavior_score"],
            "invalid_action_count": self.invalid_action_count,
            "cycle_pruning": self.cycle_pruning,
            "reverse_neighborhood_active": self.reverse_neighborhood_active,
            "reverse_neighborhood_depth": self.reverse_neighborhood_depth_used,
            **self._action_mask_info(),
            **self.simulator.counts(),
        }

        return self._get_obs(), info

    def step(self, action_idx):
        action = self.actions[action_idx]
        parts = primitive_actions(action)
        self.steps += 1

        old_fidelity = self.previous_fidelity

        gate_count = len(parts)
        t_gate_count = sum(1 for part in parts if part.name in {"T", "Tdg"})
        cnot_gate_count = sum(1 for part in parts if part.name == "CNOT")

        blocked = self.block_inverse_actions and self._would_cancel(parts)
        too_deep = self.simulator.depth + gate_count > self.max_depth
        cycle_action = False
        candidate_key = None

        if self.cycle_pruning and not blocked and not too_deep:
            candidate_key = self._unitary_key(self._candidate_unitary(action_idx))
            cycle_action = candidate_key in self.visited_unitary_keys

        invalid_action = blocked or too_deep or cycle_action

        if invalid_action:
            self.invalid_action_count += 1
            scores = self._scores()
            new_fidelity = old_fidelity
            reward = -self.invalid_action_penalty
        else:
            self.invalid_action_count = 0
            apply_action(self.simulator, action)
            self.visited_unitary_keys.add(candidate_key or self._unitary_key(self.simulator.unitary))

            scores = self._scores()
            new_fidelity = scores["behavior_score"]
            improvement = new_fidelity - old_fidelity

            reward = 10.0 * improvement

            if improvement <= 1e-8:
                reward -= self.stagnation_penalty

            reward -= self.depth_penalty * gate_count
            reward -= self.t_penalty * t_gate_count
            reward -= self.cnot_penalty * cnot_gate_count

        self.best_fidelity = max(self.best_fidelity, new_fidelity)
        unitary_fid = unitary_fidelity(self.simulator.unitary, self.target)
        self.best_unitary_fidelity = max(self.best_unitary_fidelity, unitary_fid)
        no_valid_actions = not bool(self.valid_action_mask().any())

        terminated = new_fidelity >= self.success_threshold
        truncated = (
            self.simulator.depth >= self.max_depth
            or self.steps >= self.max_steps
            or too_deep
            or no_valid_actions
            or (
                invalid_action
                and self.invalid_action_count >= self.invalid_action_patience
            )
        )

        if terminated:
            reward += 5.0

        self.previous_fidelity = new_fidelity

        info = {
            "fidelity": new_fidelity,
            "behavior_score": scores["behavior_score"],
            "basis_score": scores["basis_score"],
            "state_score": scores["state_score"],
            "observable_score": scores["observable_score"],
            "unitary_fidelity": unitary_fid,
            "best_fidelity": self.best_fidelity,
            "best_unitary_fidelity": self.best_unitary_fidelity,
            "distance": 1.0 - new_fidelity,
            "unitary_distance": 1.0 - unitary_fid,
            "action": action.name,
            "qubits": action.qubits,
            "invalid_action": invalid_action,
            "blocked_action": blocked,
            "cycle_action": cycle_action,
            "too_deep_action": too_deep,
            "no_valid_actions": no_valid_actions,
            "invalid_action_count": self.invalid_action_count,
            "circuit": self._get_circuit(),
            "simplified_depth": simplified_depth(self._get_circuit()),
            "raw_behavior_score": scores["raw_behavior_score"],
            "cycle_pruning": self.cycle_pruning,
            "reverse_neighborhood_active": self.reverse_neighborhood_active,
            "reverse_neighborhood_depth": self.reverse_neighborhood_depth_used,
            **self._action_mask_info(),
            **self.simulator.counts(),
        }

        return self._get_obs(), reward, terminated, truncated, info
