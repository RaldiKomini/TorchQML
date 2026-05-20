import json
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def load_results():
    """Load every saved experiment JSON."""
    results = []

    for path in RESULTS_DIR.glob("*.json"):
        with path.open() as f:
            data = json.load(f)

        data["file"] = path.name
        results.append(data)

    return results


def safe_stdev(values):
    """Return zero spread for single-run summaries."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def get_mean_metric(item, summary_key, detail_key):
    """Read a metric from summary fields, falling back to episode details."""
    if summary_key in item:
        return item[summary_key]

    details = item.get("episodes_detail", [])
    if not details:
        return 0.0

    return mean(detail[detail_key] for detail in details)


def first_metric(item, *keys, default=0.0):
    """Return the first metric key present in an experiment summary."""
    for key in keys:
        if key in item:
            return item[key]
    return default


def result_mode(item):
    """Infer a compact mode label from filename and summary fields."""
    name = item["file"]
    if name.startswith("mitm_"):
        return "mitm"
    if name.startswith("reverse_guided_"):
        return "reverse-guided"
    if name.startswith("ppo_distill_"):
        return "ppo-distill"
    if name.startswith("ppo_all_"):
        return str(item.get("algo", "ppo-all"))
    if name.startswith("ppo_"):
        return "ppo"
    return "random"


def summarize(results):
    """Group result files and compute table-ready aggregate metrics."""
    groups = {}

    for item in results:
        mode = result_mode(item)
        gate_set = item.get("gate_set", item.get("action_set", "base"))
        t_penalty = item.get("t_penalty", "")
        cnot_penalty = item.get("cnot_penalty", "")
        key = (
            mode,
            gate_set,
            item["target"],
            item.get("max_depth", ""),
            str(t_penalty),
            str(cnot_penalty),
        )
        groups.setdefault(tuple(str(part) for part in key), []).append(item)

    rows = []

    for (mode, gate_set, target, max_depth, t_penalty, cnot_penalty), items in sorted(groups.items()):
        success_rates = [
            first_metric(item, "success_rate", default=1.0 if item.get("success") else 0.0)
            for item in items
        ]
        fidelities = [
            first_metric(
                item,
                "mean_final_fidelity",
                "mean_unitary_fidelity",
                "mean_state_fidelity",
                "fidelity",
                default=0.0,
            )
            for item in items
        ]
        best_fidelities = [
            first_metric(
                item,
                "mean_best_fidelity",
                "mean_best_unitary_fidelity",
                "best_fidelity",
                "fidelity",
                default=get_mean_metric(item, "mean_best_fidelity", "best_fidelity"),
            )
            for item in items
        ]
        depths = [first_metric(item, "mean_depth", "depth", default=0.0) for item in items]
        simplified_depths = [
            get_mean_metric(item, "mean_simplified_depth", "simplified_depth")
            for item in items
        ]
        t_counts = [get_mean_metric(item, "mean_t_count", "t_count") for item in items]
        cnot_counts = [get_mean_metric(item, "mean_cnot_count", "cnot_count") for item in items]

        rows.append(
            {
                "mode": mode,
                "gate_set": gate_set,
                "target": target,
                "max_depth": max_depth,
                "t_penalty": t_penalty,
                "cnot_penalty": cnot_penalty,
                "runs": len(items),
                "success_mean": mean(success_rates),
                "success_std": safe_stdev(success_rates),
                "fidelity_mean": mean(fidelities),
                "fidelity_std": safe_stdev(fidelities),
                "best_fidelity_mean": mean(best_fidelities),
                "best_fidelity_std": safe_stdev(best_fidelities),
                "depth_mean": mean(depths),
                "depth_std": safe_stdev(depths),
                "simplified_depth_mean": mean(simplified_depths),
                "simplified_depth_std": safe_stdev(simplified_depths),
                "t_count_mean": mean(t_counts),
                "t_count_std": safe_stdev(t_counts),
                "cnot_count_mean": mean(cnot_counts),
                "cnot_count_std": safe_stdev(cnot_counts),
            }
        )

    return rows


def print_markdown_table(rows):
    """Print aggregate rows as a Markdown table."""
    print("| mode | gate set | target | max depth | T penalty | CNOT penalty | runs | success | fidelity | best fidelity | depth | simplified depth | T-count | CNOT-count |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['mode']} | {row['gate_set']} | {row['target']} | {row['max_depth']} | {row['t_penalty']} | {row['cnot_penalty']} | {row['runs']} | "
            f"{row['success_mean']:.3f} +/- {row['success_std']:.3f} | "
            f"{row['fidelity_mean']:.3f} +/- {row['fidelity_std']:.3f} | "
            f"{row['best_fidelity_mean']:.3f} +/- {row['best_fidelity_std']:.3f} | "
            f"{row['depth_mean']:.2f} +/- {row['depth_std']:.2f} | "
            f"{row['simplified_depth_mean']:.2f} +/- {row['simplified_depth_std']:.2f} | "
            f"{row['t_count_mean']:.2f} +/- {row['t_count_std']:.2f} | "
            f"{row['cnot_count_mean']:.2f} +/- {row['cnot_count_std']:.2f} |"
        )


def main():
    """CLI entry point for result summaries."""
    results = load_results()

    if not results:
        raise SystemExit(f"No result files found in {RESULTS_DIR}")

    rows = summarize(results)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
