import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from TorchQML.synthesis.actions_ghz import build_ghz_actions
from TorchQML.synthesis.baselines import BASELINES
from TorchQML.synthesis.env import CircuitSynthesisEnv
from TorchQML.synthesis.targets import TARGETS


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_env(target_name, max_depth=20, t_penalty=0.02, cnot_penalty=0.0, action_set="base"):
    """Create the standard training environment for one target."""
    baseline = BASELINES[target_name]
    target = TARGETS[target_name]()
    actions = None

    if action_set == "ghz":
        actions = build_ghz_actions(baseline.num_qubits)

    return CircuitSynthesisEnv(
        target=target,
        num_qubits=baseline.num_qubits,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        actions=actions,
    )


def run_random_episode(target_name, max_depth=20, t_penalty=0.02, cnot_penalty=0.0, action_set="base"):
    """Run one random-policy episode."""
    env = make_env(
        target_name,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        action_set=action_set,
    )
    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    return info


def evaluate_random(target_name, episodes, max_depth, t_penalty, cnot_penalty, action_set):
    """Evaluate the random baseline for a target."""
    results = []

    for _ in range(episodes):
        info = run_random_episode(
            target_name,
            max_depth=max_depth,
            t_penalty=t_penalty,
            cnot_penalty=cnot_penalty,
            action_set=action_set,
        )
        results.append(info)

    successes = sum(1 for item in results if item["fidelity"] >= 0.999)

    return summarize_results(
        gate_set=action_set,
        target_name=target_name,
        episodes=episodes,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        successes=successes,
        results=results,
    )


def train_ppo(target_name, seed, timesteps, max_depth, t_penalty, cnot_penalty, action_set):
    """Train a PPO policy for one target and seed."""
    env = make_env(
        target_name,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        action_set=action_set,
    )

    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_model(model, target_name, episodes, max_depth, t_penalty, cnot_penalty, action_set):
    """Run deterministic evaluation episodes for a trained policy."""
    env = make_env(
        target_name,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        action_set=action_set,
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

    successes = sum(1 for item in results if item["fidelity"] >= 0.999)

    return summarize_results(
        gate_set=action_set,
        target_name=target_name,
        episodes=episodes,
        max_depth=max_depth,
        t_penalty=t_penalty,
        cnot_penalty=cnot_penalty,
        successes=successes,
        results=results,
    )


def summarize_results(
    gate_set,
    target_name,
    episodes,
    max_depth,
    t_penalty,
    cnot_penalty,
    successes,
    results,
):
    """Pack episode metrics into the JSON result format."""
    return {
        "gate_set": gate_set,
        "target": target_name,
        "episodes": episodes,
        "max_depth": max_depth,
        "t_penalty": t_penalty,
        "cnot_penalty": cnot_penalty,
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


def penalty_label(value):
    """Make a penalty value safe for filenames."""
    return str(value).replace(".", "p")


def main():
    """CLI entry point for base/GHZ action-set experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=[*TARGETS.keys(), "all"], default="t")
    parser.add_argument("--mode", choices=["random", "ppo"], default="random")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--t-penalty", type=float, default=0.02)
    parser.add_argument("--cnot-penalty", type=float, default=0.0)
    parser.add_argument("--action-set", choices=["base", "ghz"], default="base")
    args = parser.parse_args()

    if args.target == "all":
        if args.mode != "random":
            raise SystemExit("--target all is only supported for --mode random")

        summaries = []
        for target_name in TARGETS:
            summary = evaluate_random(
                target_name=target_name,
                episodes=args.episodes,
                max_depth=args.max_depth,
                t_penalty=args.t_penalty,
                cnot_penalty=args.cnot_penalty,
                action_set=args.action_set,
            )
            output_path = (
                REPO_ROOT
                / "experiments"
                / "results"
                / f"random_{args.action_set}_{target_name}_depth{args.max_depth}_tpen{penalty_label(args.t_penalty)}_cpen{penalty_label(args.cnot_penalty)}.json"
            )
            save_json(summary, output_path)
            summaries.append(compact_summary(summary))
            print(f"saved results to {output_path}")

        print(summaries)
        return

    if args.mode == "random":
        summary = evaluate_random(
            target_name=args.target,
            episodes=args.episodes,
            max_depth=args.max_depth,
            t_penalty=args.t_penalty,
            cnot_penalty=args.cnot_penalty,
            action_set=args.action_set,
        )
        output_path = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"random_{args.action_set}_{args.target}_depth{args.max_depth}_tpen{penalty_label(args.t_penalty)}_cpen{penalty_label(args.cnot_penalty)}.json"
        )
    else:
        model = train_ppo(
            target_name=args.target,
            seed=args.seed,
            timesteps=args.timesteps,
            max_depth=args.max_depth,
            t_penalty=args.t_penalty,
            cnot_penalty=args.cnot_penalty,
            action_set=args.action_set,
        )
        summary = evaluate_model(
            model=model,
            target_name=args.target,
            episodes=args.episodes,
            max_depth=args.max_depth,
            t_penalty=args.t_penalty,
            cnot_penalty=args.cnot_penalty,
            action_set=args.action_set,
        )
        output_path = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"ppo_{args.action_set}_{args.target}_depth{args.max_depth}_tpen{penalty_label(args.t_penalty)}_cpen{penalty_label(args.cnot_penalty)}_seed{args.seed}.json"
        )

    save_json(summary, output_path)
    print(compact_summary(summary))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
