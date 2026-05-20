import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from TorchQML.synthesis.actions_pauli import build_pauli_actions
from TorchQML.synthesis.baselines import BASELINES
from TorchQML.synthesis.env import CircuitSynthesisEnv
from TorchQML.synthesis.targets import TARGETS


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_env(target_name, max_depth=20):
    """Create an environment with the Pauli-expanded action set."""
    baseline = BASELINES[target_name]
    target = TARGETS[target_name]()
    actions = build_pauli_actions(baseline.num_qubits)

    return CircuitSynthesisEnv(
        target=target,
        num_qubits=baseline.num_qubits,
        max_depth=max_depth,
        actions=actions,
    )


def train_ppo(target_name, seed, timesteps, max_depth):
    """Train PPO on the Pauli-expanded action set."""
    env = make_env(target_name, max_depth)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_model(model, target_name, episodes, max_depth):
    """Evaluate a trained Pauli-action policy."""
    env = make_env(target_name, max_depth)
    results = []

    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

        results.append(info)

    successes = sum(1 for item in results if item["fidelity"] >= 0.999)

    return {
        "gate_set": "pauli",
        "target": target_name,
        "episodes": episodes,
        "max_depth": max_depth,
        "success_rate": successes / episodes,
        "mean_final_fidelity": sum(item["fidelity"] for item in results) / episodes,
        "mean_best_fidelity": sum(item["best_fidelity"] for item in results) / episodes,
        "mean_depth": sum(item["depth"] for item in results) / episodes,
        "mean_simplified_depth": sum(item["simplified_depth"] for item in results) / episodes,
        "mean_t_count": sum(item["t_count"] for item in results) / episodes,
        "mean_cnot_count": sum(item["cnot_count"] for item in results) / episodes,
        "episodes_detail": results,
    }


def save_json(data, path):
    """Write an experiment summary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)


def compact_summary(summary):
    """Drop episode details for readable console output."""
    return {key: value for key, value in summary.items() if key != "episodes_detail"}


def main():
    """CLI entry point for Pauli-action experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS.keys(), default="toffoli")
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    args = parser.parse_args()

    model = train_ppo(
        target_name=args.target,
        seed=args.seed,
        timesteps=args.timesteps,
        max_depth=args.max_depth,
    )
    summary = evaluate_model(
        model=model,
        target_name=args.target,
        episodes=args.episodes,
        max_depth=args.max_depth,
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"ppo_pauli_{args.target}_depth{args.max_depth}_seed{args.seed}.json"
    )
    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
