import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.synthesis.baselines import BASELINES
from TorchQML.synthesis.env_distill import CircuitSynthesisDistillEnv
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


def make_env(target_name, max_depth=20, num_probe_states=64, seed=0):
    baseline = BASELINES[target_name]
    target = TARGETS[target_name]()

    probe_states = make_probe_states(
        num_states=num_probe_states,
        num_qubits=baseline.num_qubits,
        seed=seed,
    )

    return CircuitSynthesisDistillEnv(
        target=target,
        probe_states=probe_states,
        num_qubits=baseline.num_qubits,
        max_depth=max_depth,
    )


def train_ppo(target_name, seed, timesteps, max_depth, num_probe_states):
    env = make_env(
        target_name=target_name,
        max_depth=max_depth,
        num_probe_states=num_probe_states,
        seed=seed,
    )

    model = PPO("MlpPolicy", env, seed=seed, verbose=1, device="cpu")
    model.learn(total_timesteps=timesteps)

    return model


def evaluate_model(model, target_name, episodes, max_depth, num_probe_states, seed):
    env = make_env(
        target_name=target_name,
        max_depth=max_depth,
        num_probe_states=num_probe_states,
        seed=seed + 999,
    )

    results = []

    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

        results.append(info)

    successes = sum(1 for item in results if item["state_fidelity"] >= 0.999)

    return {
        "target": target_name,
        "episodes": episodes,
        "max_depth": max_depth,
        "num_probe_states": num_probe_states,
        "success_rate": successes / episodes,
        "mean_state_fidelity": sum(item["state_fidelity"] for item in results) / episodes,
        "mean_unitary_fidelity": sum(item["unitary_fidelity"] for item in results) / episodes,
        "mean_depth": sum(item["depth"] for item in results) / episodes,
        "mean_simplified_depth": sum(item["simplified_depth"] for item in results) / episodes,
        "mean_t_count": sum(item["t_count"] for item in results) / episodes,
        "mean_cnot_count": sum(item["cnot_count"] for item in results) / episodes,
        "episodes_detail": results,
    }


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)


def compact_summary(summary):
    return {key: value for key, value in summary.items() if key != "episodes_detail"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS.keys(), default="ghz")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--num-probe-states", type=int, default=64)
    args = parser.parse_args()

    model = train_ppo(
        target_name=args.target,
        seed=args.seed,
        timesteps=args.timesteps,
        max_depth=args.max_depth,
        num_probe_states=args.num_probe_states,
    )

    summary = evaluate_model(
        model=model,
        target_name=args.target,
        episodes=args.episodes,
        max_depth=args.max_depth,
        num_probe_states=args.num_probe_states,
        seed=args.seed,
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"ppo_distill_{args.target}_depth{args.max_depth}_probes{args.num_probe_states}_seed{args.seed}.json"
    )

    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
