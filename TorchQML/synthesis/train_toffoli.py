import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from TorchQML.synthesis.actions_toffoli import build_toffoli_actions
from TorchQML.synthesis.actions_toffoli_macro import build_toffoli_macro_actions
from TorchQML.synthesis.env import CircuitSynthesisEnv
from TorchQML.synthesis.targets import target_toffoli


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_env(max_depth=20, action_set="restricted"):
    """Create a Toffoli stress-test environment."""
    if action_set == "macro2":
        actions = build_toffoli_macro_actions()
        block_inverse_actions = False
    elif action_set == "blocked":
        actions = build_toffoli_actions()
        block_inverse_actions = True
    elif action_set == "restricted":
        actions = build_toffoli_actions()
        block_inverse_actions = False
    else:
        raise ValueError(f"Unknown action set: {action_set}")

    return CircuitSynthesisEnv(
        target=target_toffoli(),
        num_qubits=3,
        max_depth=max_depth,
        actions=actions,
        block_inverse_actions=block_inverse_actions,
    )


def train_ppo(seed, timesteps, max_depth, action_set):
    """Train PPO for one Toffoli action-set variant."""
    env = make_env(max_depth=max_depth, action_set=action_set)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_model(model, episodes, max_depth, action_set):
    """Evaluate a trained Toffoli policy."""
    env = make_env(max_depth=max_depth, action_set=action_set)
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
        "gate_set": f"toffoli_{action_set}",
        "target": "toffoli",
        "episodes": episodes,
        "max_depth": max_depth,
        "action_count": len(env.actions),
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
    """CLI entry point for Toffoli action-set experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument(
        "--action-set",
        choices=["restricted", "blocked", "macro2"],
        default="restricted",
    )
    args = parser.parse_args()

    model = train_ppo(
        seed=args.seed,
        timesteps=args.timesteps,
        max_depth=args.max_depth,
        action_set=args.action_set,
    )
    summary = evaluate_model(
        model=model,
        episodes=args.episodes,
        max_depth=args.max_depth,
        action_set=args.action_set,
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"ppo_toffoli_{args.action_set}_depth{args.max_depth}_seed{args.seed}.json"
    )
    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
