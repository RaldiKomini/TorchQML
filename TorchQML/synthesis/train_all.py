import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.synthesis.baselines import BASELINES
from TorchQML.synthesis.actions import build_actions
from TorchQML.synthesis.actions_toffoli import build_toffoli_actions
from TorchQML.synthesis.env_all import CircuitSynthesisAllEnv
from TorchQML.synthesis.targets import TARGETS


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_probe_states(num_states: int, num_qubits: int, seed: int = 0):
    torch.manual_seed(seed)

    dim = 1 << num_qubits
    real = torch.randn(num_states, dim, dtype=torch.float32, device=DEVICE)
    imag = torch.randn(num_states, dim, dtype=torch.float32, device=DEVICE)

    states = (real + 1j * imag).to(DTYPE)
    states = states / torch.linalg.vector_norm(states, dim=1, keepdim=True).clamp_min(1e-12)

    return states


def make_actions(num_qubits: int, action_set: str):
    if action_set == "full":
        return None

    if action_set == "minimal":
        return [
            action
            for action in build_actions(num_qubits)
            if action.name not in {"S", "Sdg"}
        ]

    if action_set == "toffoli-restricted":
        if num_qubits != 3:
            raise ValueError("toffoli-restricted action set requires 3 qubits.")
        return build_toffoli_actions()

    raise ValueError(f"Unknown action set: {action_set}")


def inverse_action_index(actions, action_idx: int) -> int:
    action = actions[action_idx]
    inverse_name = {
        "H": "H",
        "T": "Tdg",
        "Tdg": "T",
        "S": "Sdg",
        "Sdg": "S",
        "CNOT": "CNOT",
    }.get(action.name)

    if inverse_name is None:
        raise ValueError(f"No inverse action rule for {action.name}")

    inverse_key = (inverse_name, tuple(action.qubits))
    action_lookup = {
        (candidate.name, tuple(candidate.qubits)): idx
        for idx, candidate in enumerate(actions)
    }

    if inverse_key not in action_lookup:
        raise ValueError(f"Action set does not contain inverse {inverse_key}")

    return action_lookup[inverse_key]


def unitary_observation(unitary: torch.Tensor) -> torch.Tensor:
    return torch.stack([unitary.real, unitary.imag]).to(dtype=torch.float32)


def make_env(
    target_name,
    max_depth=20,
    num_probe_states=64,
    seed=0,
    invalid_action_penalty=5.0,
    invalid_action_patience=1,
    action_set="full",
    cycle_pruning=True,
    reverse_neighborhood_probability=0.0,
    reverse_neighborhood_depth=0,
):
    baseline = BASELINES[target_name]
    target = TARGETS[target_name]()
    actions = make_actions(baseline.num_qubits, action_set)

    probe_states = make_probe_states(
        num_states=num_probe_states,
        num_qubits=baseline.num_qubits,
        seed=seed,
    )

    return CircuitSynthesisAllEnv(
        target=target,
        probe_states=probe_states,
        num_qubits=baseline.num_qubits,
        max_depth=max_depth,
        invalid_action_penalty=invalid_action_penalty,
        invalid_action_patience=invalid_action_patience,
        actions=actions,
        cycle_pruning=cycle_pruning,
        reverse_neighborhood_probability=reverse_neighborhood_probability,
        reverse_neighborhood_depth=reverse_neighborhood_depth,
    )


def make_reverse_pretrain_batch(env, batch_size: int, max_depth: int, generator):
    observations = []
    labels = []

    for _ in range(batch_size):
        unitary = env.target.clone()
        depth = int(torch.randint(1, max_depth + 1, (1,), generator=generator).item())
        last_action_idx = 0

        for _ in range(depth):
            action_idx = int(
                torch.randint(
                    0,
                    len(env.action_unitaries),
                    (1,),
                    generator=generator,
                ).item()
            )
            unitary = env.action_unitaries[action_idx] @ unitary
            last_action_idx = action_idx

        observations.append(unitary_observation(unitary))
        labels.append(inverse_action_index(env.actions, last_action_idx))

    return (
        torch.stack(observations),
        torch.tensor(labels, dtype=torch.long),
    )


def policy_logits(policy, observations):
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        policy_features = features[0]
    else:
        policy_features = features

    latent_pi, _ = policy.mlp_extractor(policy_features)
    return policy.action_net(latent_pi)


def pretrain_reverse_neighborhood_policy(
    model,
    env,
    seed: int,
    steps: int,
    batch_size: int,
    max_depth: int,
    learning_rate: float,
):
    if steps <= 0:
        return {"pretrain_steps": 0}

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    device = model.policy.device
    last_loss = None
    last_accuracy = None

    model.policy.train()

    for _ in range(steps):
        observations, labels = make_reverse_pretrain_batch(
            env=env,
            batch_size=batch_size,
            max_depth=max_depth,
            generator=generator,
        )
        observations = observations.to(device)
        labels = labels.to(device)

        logits = policy_logits(model.policy, observations)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        predictions = logits.argmax(dim=1)
        last_accuracy = float((predictions == labels).float().mean().detach().cpu().item())

    model.policy.eval()

    return {
        "pretrain_steps": steps,
        "pretrain_batch_size": batch_size,
        "pretrain_max_depth": max_depth,
        "pretrain_learning_rate": learning_rate,
        "pretrain_final_loss": last_loss,
        "pretrain_final_accuracy": last_accuracy,
    }


def train_model(
    target_name,
    seed,
    timesteps,
    max_depth,
    num_probe_states,
    invalid_action_penalty,
    invalid_action_patience,
    algo,
    action_set,
    cycle_pruning,
    reverse_neighborhood_probability,
    reverse_neighborhood_depth,
    pretrain_steps,
    pretrain_batch_size,
    pretrain_depth,
    pretrain_learning_rate,
):
    env = make_env(
        target_name=target_name,
        max_depth=max_depth,
        num_probe_states=num_probe_states,
        seed=seed,
        invalid_action_penalty=invalid_action_penalty,
        invalid_action_patience=invalid_action_patience,
        action_set=action_set,
        cycle_pruning=cycle_pruning,
        reverse_neighborhood_probability=reverse_neighborhood_probability,
        reverse_neighborhood_depth=reverse_neighborhood_depth,
    )

    model_cls = MaskablePPO if algo == "maskableppo" else PPO
    model = model_cls("MlpPolicy", env, seed=seed, verbose=1, device="cpu")
    pretrain_info = pretrain_reverse_neighborhood_policy(
        model=model,
        env=env,
        seed=seed + 12345,
        steps=pretrain_steps,
        batch_size=pretrain_batch_size,
        max_depth=pretrain_depth,
        learning_rate=pretrain_learning_rate,
    )
    if pretrain_info["pretrain_steps"]:
        print(pretrain_info)
    model.learn(total_timesteps=timesteps)

    return model, pretrain_info


def train_ppo(target_name, seed, timesteps, max_depth, num_probe_states):
    model, _ = train_model(
        target_name=target_name,
        seed=seed,
        timesteps=timesteps,
        max_depth=max_depth,
        num_probe_states=num_probe_states,
        invalid_action_penalty=5.0,
        invalid_action_patience=1,
        algo="ppo",
        action_set="full",
        cycle_pruning=True,
        reverse_neighborhood_probability=0.0,
        reverse_neighborhood_depth=0,
        pretrain_steps=0,
        pretrain_batch_size=256,
        pretrain_depth=6,
        pretrain_learning_rate=1e-3,
    )
    return model


def evaluate_model(
    model,
    target_name,
    episodes,
    max_depth,
    num_probe_states,
    seed,
    invalid_action_penalty,
    invalid_action_patience,
    algo,
    action_set,
    cycle_pruning,
):
    env = make_env(
        target_name=target_name,
        max_depth=max_depth,
        num_probe_states=num_probe_states,
        seed=seed + 999,
        invalid_action_penalty=invalid_action_penalty,
        invalid_action_patience=invalid_action_patience,
        action_set=action_set,
        cycle_pruning=cycle_pruning,
        reverse_neighborhood_probability=0.0,
        reverse_neighborhood_depth=0,
    )

    results = []

    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            predict_kwargs = {}
            if algo == "maskableppo":
                predict_kwargs["action_masks"] = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, **predict_kwargs)
            obs, reward, terminated, truncated, info = env.step(int(action))

        results.append(info)

    successes = sum(1 for item in results if item["unitary_fidelity"] >= 0.999)

    return {
        "target": target_name,
        "episodes": episodes,
        "max_depth": max_depth,
        "num_probe_states": num_probe_states,
        "algo": algo,
        "action_set": action_set,
        "action_count": len(env.actions),
        "cycle_pruning": cycle_pruning,
        "invalid_action_penalty": invalid_action_penalty,
        "invalid_action_patience": invalid_action_patience,
        "success_rate": successes / episodes,

        "mean_behavior_score": sum(item["behavior_score"] for item in results) / episodes,
        "mean_basis_score": sum(item["basis_score"] for item in results) / episodes,
        "mean_state_score": sum(item["state_score"] for item in results) / episodes,
        "mean_observable_score": sum(item["observable_score"] for item in results) / episodes,
        "mean_unitary_fidelity": sum(item["unitary_fidelity"] for item in results) / episodes,
        "mean_best_unitary_fidelity": sum(item["best_unitary_fidelity"] for item in results) / episodes,

        "mean_depth": sum(item["depth"] for item in results) / episodes,
        "mean_simplified_depth": sum(item["simplified_depth"] for item in results) / episodes,
        "mean_t_count": sum(item["t_count"] for item in results) / episodes,
        "mean_cnot_count": sum(item["cnot_count"] for item in results) / episodes,
        "mean_invalid_action_count": sum(item["invalid_action_count"] for item in results) / episodes,

        "episodes_detail": results,
    }


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)


def compact_summary(summary):
    return {key: value for key, value in summary.items() if key != "episodes_detail"}


def probability_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS.keys(), default="ghz")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--num-probe-states", type=int, default=64)
    parser.add_argument("--invalid-action-penalty", type=float, default=5.0)
    parser.add_argument("--invalid-action-patience", type=int, default=1)
    parser.add_argument("--algo", choices=["maskableppo", "ppo"], default="maskableppo")
    parser.add_argument(
        "--action-set",
        choices=["full", "minimal", "toffoli-restricted"],
        default="full",
    )
    parser.add_argument("--no-cycle-pruning", action="store_true")
    parser.add_argument("--reverse-neighborhood-probability", type=float, default=0.0)
    parser.add_argument("--reverse-neighborhood-depth", type=int, default=0)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-depth", type=int, default=6)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    cycle_pruning = not args.no_cycle_pruning

    model, pretrain_info = train_model(
        target_name=args.target,
        seed=args.seed,
        timesteps=args.timesteps,
        max_depth=args.max_depth,
        num_probe_states=args.num_probe_states,
        invalid_action_penalty=args.invalid_action_penalty,
        invalid_action_patience=args.invalid_action_patience,
        algo=args.algo,
        action_set=args.action_set,
        cycle_pruning=cycle_pruning,
        reverse_neighborhood_probability=args.reverse_neighborhood_probability,
        reverse_neighborhood_depth=args.reverse_neighborhood_depth,
        pretrain_steps=args.pretrain_steps,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_depth=args.pretrain_depth,
        pretrain_learning_rate=args.pretrain_learning_rate,
    )

    summary = evaluate_model(
        model=model,
        target_name=args.target,
        episodes=args.episodes,
        max_depth=args.max_depth,
        num_probe_states=args.num_probe_states,
        seed=args.seed,
        invalid_action_penalty=args.invalid_action_penalty,
        invalid_action_patience=args.invalid_action_patience,
        algo=args.algo,
        action_set=args.action_set,
        cycle_pruning=cycle_pruning,
    )
    summary["reverse_neighborhood_probability"] = args.reverse_neighborhood_probability
    summary["reverse_neighborhood_depth"] = args.reverse_neighborhood_depth
    summary.update(pretrain_info)

    reverse_label = (
        f"_revp{probability_label(args.reverse_neighborhood_probability)}"
        f"_revd{args.reverse_neighborhood_depth}"
    )
    pretrain_label = (
        f"_pre{args.pretrain_steps}"
        f"_pred{args.pretrain_depth}"
    )
    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / (
            f"{args.algo}_all_{args.action_set}_{args.target}"
            f"_depth{args.max_depth}_probes{args.num_probe_states}"
            f"{reverse_label}{pretrain_label}_cycle{int(cycle_pruning)}_seed{args.seed}.json"
        )
    )

    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
