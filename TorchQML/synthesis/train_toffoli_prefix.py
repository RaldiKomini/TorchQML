import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from TorchQML.synthesis.actions_toffoli import build_toffoli_actions
from TorchQML.synthesis.env import CircuitSynthesisEnv
from TorchQML.synthesis.targets import target_toffoli_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFIX_DEPTHS = [3, 5, 7, 9, 11, 13, 15]


def make_env(start_depth, target_depth, max_depth, block_inverse):
    """Create one oracle-start Toffoli prefix task."""
    return CircuitSynthesisEnv(
        target=target_toffoli_prefix(target_depth),
        start_unitary=target_toffoli_prefix(start_depth),
        num_qubits=3,
        max_depth=max_depth,
        actions=build_toffoli_actions(),
        block_inverse_actions=block_inverse,
    )


def train_ppo(seed, timesteps, start_depth, target_depth, max_depth, block_inverse):
    """Train PPO for one prefix continuation task."""
    env = make_env(start_depth, target_depth, max_depth, block_inverse)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_model(model, episodes, start_depth, target_depth, max_depth, block_inverse, mode):
    """Evaluate one trained prefix policy."""
    env = make_env(start_depth, target_depth, max_depth, block_inverse)
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
        "gate_set": "toffoli_prefix",
        "target": f"toffoli_prefix_{target_depth}",
        "probe_mode": mode,
        "start_depth": start_depth,
        "target_depth": target_depth,
        "known_added_depth": target_depth - start_depth,
        "episodes": episodes,
        "max_depth": max_depth,
        "action_count": len(env.actions),
        "block_inverse_actions": block_inverse,
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


def run_prefix_task(args, start_depth, target_depth):
    """Train, evaluate, and save one prefix task."""
    if args.fixed_max_depth is None:
        max_depth = target_depth - start_depth + args.extra_depth
    else:
        max_depth = args.fixed_max_depth
    model = train_ppo(
        seed=args.seed,
        timesteps=args.timesteps,
        start_depth=start_depth,
        target_depth=target_depth,
        max_depth=max_depth,
        block_inverse=args.block_inverse,
    )
    summary = evaluate_model(
        model=model,
        episodes=args.episodes,
        start_depth=start_depth,
        target_depth=target_depth,
        max_depth=max_depth,
        block_inverse=args.block_inverse,
        mode=args.mode,
    )

    output_path = (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"ppo_toffoli_prefix_{args.mode}_start{start_depth}_target{target_depth}_max{max_depth}_seed{args.seed}.json"
    )
    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


def main():
    """CLI entry point for Toffoli prefix experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["identity", "continuation"], default="identity")
    parser.add_argument("--prefix-depths", nargs="+", type=int, default=PREFIX_DEPTHS)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--extra-depth", type=int, default=0)
    parser.add_argument("--fixed-max-depth", type=int, default=None)
    parser.add_argument("--block-inverse", action="store_true")
    args = parser.parse_args()

    start_depth = 0

    for target_depth in args.prefix_depths:
        if target_depth <= start_depth:
            raise ValueError("prefix depths must increase")

        task_start = 0 if args.mode == "identity" else start_depth
        run_prefix_task(args, task_start, target_depth)
        start_depth = target_depth


if __name__ == "__main__":
    main()
