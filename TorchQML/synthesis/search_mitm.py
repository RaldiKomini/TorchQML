import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.synthesis.metrics import unitary_fidelity
from TorchQML.synthesis.train_all import make_env, inverse_action_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def unitary_key(unitary: np.ndarray, decimals: int) -> bytes:
    canonical = unitary
    flat = canonical.reshape(-1)
    nonzero = np.abs(flat) > 1e-8

    if np.any(nonzero):
        pivot = flat[nonzero][0]
        canonical = canonical / (pivot / abs(pivot))

    real_imag = np.stack([canonical.real, canonical.imag], axis=-1)
    return np.round(real_imag, decimals=decimals).astype(np.float32, copy=False).tobytes()


def append_code(code: int, action_idx: int, base: int) -> int:
    return code * base + action_idx


def decode_code(code: int, depth: int, base: int) -> tuple[int, ...]:
    path = [0] * depth

    for idx in range(depth - 1, -1, -1):
        path[idx] = code % base
        code //= base

    return tuple(path)


def invert_path(env, path: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(inverse_action_index(env.actions, idx) for idx in reversed(path))


def circuit_from_path(
    env,
    path: tuple[int, ...],
    original_action_indices: list[int] | None = None,
) -> list[dict]:
    return [
        {
            "index": idx,
            "original_index": (
                original_action_indices[idx]
                if original_action_indices is not None
                else idx
            ),
            "name": env.actions[idx].name,
            "qubits": list(env.actions[idx].qubits),
        }
        for idx in path
    ]


def verify_path(env, path: tuple[int, ...]) -> float:
    unitary = torch.eye(env.dim, dtype=DTYPE, device=DEVICE)

    for idx in path:
        unitary = env.action_unitaries[idx] @ unitary

    return unitary_fidelity(unitary, env.target)


def basis_bit(index: int, num_qubits: int, qubit: int) -> int:
    return (index >> qubit) & 1


def monomial_permutation(target: torch.Tensor, tolerance: float) -> np.ndarray | None:
    matrix = target.detach().cpu().numpy()
    magnitudes = np.abs(matrix)
    rows = np.argmax(magnitudes, axis=0)

    if not np.all(magnitudes[rows, np.arange(matrix.shape[1])] >= 1.0 - tolerance):
        return None

    support = magnitudes > tolerance
    if not np.all(support.sum(axis=0) == 1):
        return None
    if not np.all(support.sum(axis=1) == 1):
        return None
    if len(set(rows.tolist())) != matrix.shape[0]:
        return None

    return rows


def is_affine_boolean(values: list[int], num_qubits: int) -> bool:
    constant = values[0]
    coefficients = [
        values[1 << qubit] ^ constant
        for qubit in range(num_qubits)
    ]

    for basis, value in enumerate(values):
        predicted = constant
        for qubit, coefficient in enumerate(coefficients):
            predicted ^= coefficient & basis_bit(basis, num_qubits, qubit)

        if predicted != value:
            return False

    return True


def non_affine_output_qubits(permutation: np.ndarray, num_qubits: int) -> list[int]:
    qubits = []

    for output_qubit in range(num_qubits):
        values = [
            basis_bit(int(permutation[input_basis]), num_qubits, output_qubit)
            for input_basis in range(len(permutation))
        ]
        if not is_affine_boolean(values, num_qubits):
            qubits.append(output_qubit)

    return qubits


def apply_action_pruning(env, pruning: str, tolerance: float):
    original_actions = list(env.actions)
    original_unitaries = list(env.action_unitaries)
    original_indices = list(range(len(original_actions)))
    metadata = {
        "action_pruning": pruning,
        "original_action_count": len(original_actions),
        "selected_action_indices": original_indices,
        "selected_h_qubits": None,
        "monomial_target": None,
    }

    if pruning == "none":
        return metadata, original_indices

    if pruning != "monomial-h":
        raise ValueError(f"Unknown action pruning mode: {pruning}")

    permutation = monomial_permutation(env.target, tolerance)
    if permutation is None:
        metadata["monomial_target"] = False
        return metadata, original_indices

    selected_h_qubits = non_affine_output_qubits(permutation, env.num_qubits)
    selected = [
        idx
        for idx, action in enumerate(original_actions)
        if action.name != "H" or action.qubits[0] in selected_h_qubits
    ]

    env.actions = [original_actions[idx] for idx in selected]
    env.action_unitaries = [original_unitaries[idx] for idx in selected]
    metadata.update(
        {
            "selected_action_indices": selected,
            "selected_h_qubits": selected_h_qubits,
            "monomial_target": True,
        }
    )

    return metadata, selected


def expand_frontier(
    *,
    env,
    actions_np: list[np.ndarray],
    frontier: list[tuple[np.ndarray, int, int, int]],
    seen,
    base: int,
    decimals: int,
    skip_immediate_inverse: bool,
) -> list[tuple[np.ndarray, int, int, int]]:
    new_frontier = []

    for unitary, code, depth, last_action in frontier:
        inverse_last = (
            inverse_action_index(env.actions, last_action)
            if skip_immediate_inverse and last_action >= 0
            else None
        )

        for action_idx, action_unitary in enumerate(actions_np):
            if inverse_last is not None and action_idx == inverse_last:
                continue

            candidate = action_unitary @ unitary
            key = unitary_key(candidate, decimals)
            if key in seen:
                continue

            next_code = append_code(code, action_idx, base)
            next_depth = depth + 1
            seen[key] = (next_depth, next_code)
            new_frontier.append((candidate, next_code, next_depth, action_idx))

    return new_frontier


def build_forward_table(env, actions_np, args):
    identity = np.eye(env.dim, dtype=np.complex64)
    base = len(actions_np)
    start_key = unitary_key(identity, args.decimals)
    seen = {start_key: (0, 0)}
    frontier = [(identity, 0, 0, -1)]
    started_at = time.time()

    for depth in range(1, args.forward_depth + 1):
        step_started_at = time.time()
        frontier = expand_frontier(
            env=env,
            actions_np=actions_np,
            frontier=frontier,
            seen=seen,
            base=base,
            decimals=args.decimals,
            skip_immediate_inverse=args.skip_immediate_inverse,
        )
        print(
            {
                "side": "forward",
                "depth": depth,
                "frontier": len(frontier),
                "states": len(seen),
                "seconds": round(time.time() - step_started_at, 3),
            },
            flush=True,
        )

        if args.max_forward_states and len(seen) > args.max_forward_states:
            raise RuntimeError(
                f"forward table exceeded --max-forward-states={args.max_forward_states}"
            )

        if not frontier:
            break

    return seen, time.time() - started_at


def search_backward(env, actions_np, forward_table, args):
    target = env.target.detach().cpu().numpy().astype(np.complex64)
    base = len(actions_np)
    target_key = unitary_key(target, args.decimals)
    seen = {target_key: (0, 0)}
    frontier = [(target, 0, 0, -1)]
    started_at = time.time()
    rejected_meets = 0

    for depth in range(0, args.backward_depth + 1):
        for unitary, code, node_depth, _ in frontier:
            key = unitary_key(unitary, args.decimals)
            if key not in forward_table:
                continue

            forward_depth, forward_code = forward_table[key]
            forward_path = decode_code(forward_code, forward_depth, base)
            backward_path = decode_code(code, node_depth, base)
            full_path = (*forward_path, *invert_path(env, backward_path))
            fidelity = verify_path(env, full_path)
            if fidelity >= args.success_threshold:
                return {
                    "success": True,
                    "path": full_path,
                    "fidelity": fidelity,
                    "depth": len(full_path),
                    "meet_forward_depth": forward_depth,
                    "meet_backward_depth": node_depth,
                    "backward_states": len(seen),
                    "backward_seconds": time.time() - started_at,
                    "rejected_meets": rejected_meets,
                }

            rejected_meets += 1

        if depth == args.backward_depth:
            break

        step_started_at = time.time()
        frontier = expand_frontier(
            env=env,
            actions_np=actions_np,
            frontier=frontier,
            seen=seen,
            base=base,
            decimals=args.decimals,
            skip_immediate_inverse=args.skip_immediate_inverse,
        )
        print(
            {
                "side": "backward",
                "depth": depth + 1,
                "frontier": len(frontier),
                "states": len(seen),
                "seconds": round(time.time() - step_started_at, 3),
            },
            flush=True,
        )

        if args.max_backward_states and len(seen) > args.max_backward_states:
            raise RuntimeError(
                f"backward search exceeded --max-backward-states={args.max_backward_states}"
            )

        if not frontier:
            break

    return {
        "success": False,
        "path": (),
        "fidelity": None,
        "depth": None,
        "meet_forward_depth": None,
        "meet_backward_depth": None,
        "backward_states": len(seen),
        "backward_seconds": time.time() - started_at,
        "rejected_meets": rejected_meets,
    }


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="toffoli")
    parser.add_argument(
        "--action-set",
        default="minimal",
        choices=["full", "minimal", "toffoli-restricted"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-probe-states", type=int, default=8)
    parser.add_argument("--forward-depth", type=int, default=7)
    parser.add_argument("--backward-depth", type=int, default=8)
    parser.add_argument("--decimals", type=int, default=5)
    parser.add_argument(
        "--action-pruning",
        default="none",
        choices=["none", "monomial-h"],
    )
    parser.add_argument("--monomial-tolerance", type=float, default=1e-5)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--max-forward-states", type=int, default=0)
    parser.add_argument("--max-backward-states", type=int, default=0)
    parser.add_argument(
        "--no-skip-immediate-inverse",
        dest="skip_immediate_inverse",
        action="store_false",
    )
    parser.set_defaults(skip_immediate_inverse=True)
    args = parser.parse_args()

    env = make_env(
        target_name=args.target,
        max_depth=args.forward_depth + args.backward_depth,
        num_probe_states=args.num_probe_states,
        seed=args.seed,
        action_set=args.action_set,
        cycle_pruning=True,
    )
    pruning_metadata, original_action_indices = apply_action_pruning(
        env,
        args.action_pruning,
        args.monomial_tolerance,
    )
    actions_np = [
        unitary.detach().cpu().numpy().astype(np.complex64)
        for unitary in env.action_unitaries
    ]

    forward_table, forward_seconds = build_forward_table(env, actions_np, args)
    result = search_backward(env, actions_np, forward_table, args)
    result.update(
        {
            "target": args.target,
            "action_set": args.action_set,
            "action_count": len(env.actions),
            **pruning_metadata,
            "pruned_action_count": len(env.actions),
            "forward_depth": args.forward_depth,
            "backward_depth": args.backward_depth,
            "forward_states": len(forward_table),
            "forward_seconds": forward_seconds,
            "skip_immediate_inverse": args.skip_immediate_inverse,
            "circuit": circuit_from_path(
                env,
                result["path"],
                original_action_indices=original_action_indices,
            ),
        }
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / (
            f"mitm_{args.action_set}_{args.target}"
            f"_{args.action_pruning}"
            f"_f{args.forward_depth}_b{args.backward_depth}_seed{args.seed}.json"
        )
    )
    save_json(result, output_path)
    print(result, flush=True)
    print(f"saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
