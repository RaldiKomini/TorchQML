import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.synthesis.metrics import unitary_fidelity
from TorchQML.synthesis.train_all import make_env, inverse_action_index


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReverseGuide(nn.Module):
    def __init__(self, dim: int, action_count: int):
        super().__init__()
        input_dim = 2 * dim * dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(128, action_count)
        self.distance_head = nn.Linear(128, 1)

    def forward(self, observations):
        features = self.net(observations.flatten(start_dim=1))
        return self.action_head(features), self.distance_head(features).squeeze(-1)


@dataclass
class Node:
    unitary: torch.Tensor
    path: tuple[int, ...]
    score: float
    fidelity: float
    predicted_distance: float


def unitary_key(unitary: torch.Tensor, decimals: int = 5) -> bytes:
    canonical = unitary.detach().cpu()
    flat = canonical.flatten()
    nonzero = flat.abs() > 1e-8

    if bool(nonzero.any()):
        pivot = flat[nonzero][0]
        canonical = canonical / (pivot / pivot.abs())

    rounded = torch.round(torch.view_as_real(canonical), decimals=decimals)
    return rounded.numpy().tobytes()


def unitary_observation(unitary: torch.Tensor) -> torch.Tensor:
    return torch.stack([unitary.real, unitary.imag]).to(dtype=torch.float32)


def relative_observation(unitary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return unitary_observation(unitary @ target.conj().T)


def make_batch(env, batch_size: int, max_depth: int, generator):
    observations = []
    action_labels = []
    distance_labels = []

    for _ in range(batch_size):
        unitary = env.target.clone()
        depth = int(torch.randint(1, max_depth + 1, (1,), generator=generator).item())
        last_action_idx = 0

        for _ in range(depth):
            action_idx = int(
                torch.randint(0, len(env.action_unitaries), (1,), generator=generator).item()
            )
            unitary = env.action_unitaries[action_idx] @ unitary
            last_action_idx = action_idx

        observations.append(relative_observation(unitary, env.target))
        action_labels.append(inverse_action_index(env.actions, last_action_idx))
        distance_labels.append(float(depth))

    return (
        torch.stack(observations),
        torch.tensor(action_labels, dtype=torch.long),
        torch.tensor(distance_labels, dtype=torch.float32),
    )


def train_guide(env, args):
    model = ReverseGuide(env.dim, len(env.actions)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 2026)
    last = {}

    model.train()
    for step in range(1, args.train_steps + 1):
        observations, action_labels, distance_labels = make_batch(
            env,
            args.train_batch_size,
            args.train_depth,
            generator,
        )
        observations = observations.to(DEVICE)
        action_labels = action_labels.to(DEVICE)
        distance_labels = distance_labels.to(DEVICE)

        action_logits, distance_pred = model(observations)
        action_loss = nn.functional.cross_entropy(action_logits, action_labels)
        distance_loss = nn.functional.smooth_l1_loss(
            distance_pred / args.train_depth,
            distance_labels / args.train_depth,
        )
        loss = action_loss + args.distance_loss_weight * distance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == args.train_steps or step % args.report_every == 0:
            with torch.no_grad():
                accuracy = (action_logits.argmax(dim=1) == action_labels).float().mean()
            last = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "action_loss": float(action_loss.detach().cpu().item()),
                "distance_loss": float(distance_loss.detach().cpu().item()),
                "action_accuracy": float(accuracy.detach().cpu().item()),
            }
            print({"guide_train": last})

    model.eval()
    return model, last


def evaluate_node(model, env, unitary, path, args):
    fidelity = unitary_fidelity(unitary, env.target)
    observation = relative_observation(unitary, env.target).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action_logits, distance_pred = model(observation)

    predicted_distance = float(distance_pred.squeeze(0).detach().cpu().item())
    path_penalty = args.path_penalty * len(path)
    fidelity_bonus = args.fidelity_weight * fidelity
    score = predicted_distance + path_penalty - fidelity_bonus

    return Node(
        unitary=unitary,
        path=path,
        score=score,
        fidelity=fidelity,
        predicted_distance=predicted_distance,
    )


def top_actions(model, env, unitary, args):
    observation = relative_observation(unitary, env.target).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(observation)

    count = min(args.top_actions, len(env.actions))
    learned = torch.topk(logits.squeeze(0), k=count).indices.detach().cpu().tolist()

    if args.include_all_actions:
        return list(dict.fromkeys([*learned, *range(len(env.actions))]))

    return learned


def invert_path(env, path):
    return tuple(inverse_action_index(env.actions, idx) for idx in reversed(path))


def build_backward_table(model, env, args):
    start = evaluate_node(model, env, env.target.clone(), (), args)
    beam = [start]
    table = {unitary_key(start.unitary): ()}
    seen = {unitary_key(start.unitary)}

    for _ in range(args.backward_depth):
        candidates = []

        for node in beam:
            for action_idx in top_actions(model, env, node.unitary, args):
                unitary = env.action_unitaries[action_idx] @ node.unitary
                key = unitary_key(unitary)
                if key in seen:
                    continue

                seen.add(key)
                path = (*node.path, action_idx)
                table.setdefault(key, path)
                candidates.append(evaluate_node(model, env, unitary, path, args))

        candidates.sort(key=lambda node: (node.score, -node.fidelity, len(node.path)))
        beam = candidates[: args.backward_beam_width]

        if not beam:
            break

    return table


def search(model, env, args):
    identity = torch.eye(env.dim, dtype=DTYPE, device=DEVICE)
    start = evaluate_node(model, env, identity, (), args)
    beam = [start]
    seen = {unitary_key(identity)}
    best = start
    backward_table = build_backward_table(model, env, args)

    for depth in range(args.search_depth):
        candidates = []

        for node in beam:
            for action_idx in top_actions(model, env, node.unitary, args):
                unitary = env.action_unitaries[action_idx] @ node.unitary
                key = unitary_key(unitary)
                if key in seen:
                    continue

                path = (*node.path, action_idx)

                if key in backward_table:
                    full_path = (*path, *invert_path(env, backward_table[key]))
                    full_unitary = identity.clone()
                    for idx in full_path:
                        full_unitary = env.action_unitaries[idx] @ full_unitary
                    fidelity = unitary_fidelity(full_unitary, env.target)
                    if fidelity >= args.success_threshold:
                        return {
                            "success": True,
                            "path": full_path,
                            "fidelity": fidelity,
                            "depth": len(full_path),
                            "meet_depth": depth + 1,
                            "backward_depth": len(backward_table[key]),
                            "best_fidelity": max(best.fidelity, fidelity),
                        }

                seen.add(key)
                candidate = evaluate_node(model, env, unitary, path, args)
                candidates.append(candidate)

                if candidate.fidelity > best.fidelity:
                    best = candidate

                if candidate.fidelity >= args.success_threshold:
                    return {
                        "success": True,
                        "path": candidate.path,
                        "fidelity": candidate.fidelity,
                        "depth": len(candidate.path),
                        "meet_depth": None,
                        "backward_depth": 0,
                        "best_fidelity": candidate.fidelity,
                    }

        candidates.sort(key=lambda node: (node.score, -node.fidelity, len(node.path)))
        beam = candidates[: args.beam_width]

        print(
            {
                "search_depth": depth + 1,
                "beam": len(beam),
                "seen": len(seen),
                "best_fidelity": best.fidelity,
                "best_depth": len(best.path),
                "best_predicted_distance": best.predicted_distance,
            }
        )

        if not beam:
            break

    return {
        "success": False,
        "path": best.path,
        "fidelity": best.fidelity,
        "depth": len(best.path),
        "meet_depth": None,
        "backward_depth": 0,
        "best_fidelity": best.fidelity,
    }


def circuit_from_path(env, path):
    return [
        {
            "index": idx,
            "name": env.actions[idx].name,
            "qubits": list(env.actions[idx].qubits),
        }
        for idx in path
    ]


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="toffoli")
    parser.add_argument("--action-set", default="minimal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-probe-states", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--train-depth", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--distance-loss-weight", type=float, default=2.0)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--beam-width", type=int, default=1000)
    parser.add_argument("--backward-beam-width", type=int, default=1000)
    parser.add_argument("--backward-depth", type=int, default=5)
    parser.add_argument("--search-depth", type=int, default=20)
    parser.add_argument("--top-actions", type=int, default=6)
    parser.add_argument("--include-all-actions", action="store_true")
    parser.add_argument("--path-penalty", type=float, default=0.03)
    parser.add_argument("--fidelity-weight", type=float, default=4.0)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = make_env(
        target_name=args.target,
        max_depth=max(args.search_depth + args.backward_depth, 1),
        num_probe_states=args.num_probe_states,
        seed=args.seed,
        action_set=args.action_set,
        cycle_pruning=True,
    )
    model, train_info = train_guide(env, args)
    result = search(model, env, args)
    result.update(
        {
            "target": args.target,
            "action_set": args.action_set,
            "action_count": len(env.actions),
            "train_info": train_info,
            "beam_width": args.beam_width,
            "backward_beam_width": args.backward_beam_width,
            "backward_depth": args.backward_depth,
            "search_depth": args.search_depth,
            "top_actions": args.top_actions,
            "include_all_actions": args.include_all_actions,
            "circuit": circuit_from_path(env, result["path"]),
        }
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / (
            f"reverse_guided_{args.action_set}_{args.target}"
            f"_beam{args.beam_width}_bwd{args.backward_depth}"
            f"_train{args.train_steps}_seed{args.seed}.json"
        )
    )
    save_json(result, output_path)
    print(result)
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
