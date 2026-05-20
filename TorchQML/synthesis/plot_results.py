import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from TorchQML.core.unitary import UnitarySimulator
from TorchQML.synthesis.metrics import unitary_fidelity
from TorchQML.synthesis.targets import target_toffoli_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
PLOTS_DIR = REPO_ROOT / "experiments" / "plots"


COLORS = {
    "blue": "#3b6ea8",
    "orange": "#e68632",
    "green": "#4c9a61",
    "red": "#c75146",
    "gray": "#6f7782",
    "dark": "#222831",
}


def setup_style():
    """Set a small, consistent style for report figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c7cbd1",
            "axes.labelcolor": COLORS["dark"],
            "axes.titlecolor": COLORS["dark"],
            "axes.grid": True,
            "grid.color": "#e6e8eb",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "legend.frameon": False,
        }
    )


def load_results(pattern: str) -> list[dict]:
    """Load saved result files matching a glob pattern."""
    results = []

    for path in sorted(RESULTS_DIR.glob(pattern)):
        with path.open() as f:
            item = json.load(f)
        item["file"] = path.name
        results.append(item)

    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, with empty data handled."""
    if len(values) == 0:
        return np.nan, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def summarize_metric(pattern: str, key: str) -> tuple[float, float]:
    """Aggregate one metric over files matching a pattern."""
    values = [item[key] for item in load_results(pattern)]
    return mean_std(values)


def save(fig, filename: str):
    """Save a Matplotlib figure into the experiment plots folder."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axes(ax):
    """Remove visual clutter from a plot axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def annotate_bars(ax, bars, values):
    """Write small numeric labels above bars."""
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["dark"],
        )


def plot_ghz_action_space():
    """Plot the GHZ base-vs-restricted action space comparison."""
    labels = ["Base\n21 actions", "GHZ-only\n9 actions"]
    patterns = [
        "ppo_base_ghz_depth15_tpen0p02_cpen0p0_seed*.json",
        "ppo_ghz_ghz_depth15_tpen0p02_cpen0p0_seed*.json",
    ]

    success = [summarize_metric(pattern, "success_rate") for pattern in patterns]
    fidelity = [summarize_metric(pattern, "mean_final_fidelity") for pattern in patterns]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_a = ax.bar(
        x - width / 2,
        [value for value, _ in success],
        width,
        yerr=[err for _, err in success],
        capsize=4,
        label="success rate",
        color=COLORS["blue"],
    )
    bars_b = ax.bar(
        x + width / 2,
        [value for value, _ in fidelity],
        width,
        yerr=[err for _, err in fidelity],
        capsize=4,
        label="final fidelity",
        color=COLORS["green"],
    )

    ax.set_title("GHZ: removing irrelevant gates makes PPO reliable")
    ax.set_ylabel("mean over 3 seeds")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    clean_axes(ax)
    annotate_bars(ax, bars_a, [value for value, _ in success])
    annotate_bars(ax, bars_b, [value for value, _ in fidelity])
    save(fig, "figure_ghz_action_space.png")


def plot_toffoli_from_identity():
    """Plot Toffoli prefix synthesis with the identity baseline removed."""
    prefix_depths = [3, 5, 7, 9, 11, 13, 15]
    success = []
    improvement = []
    identity = UnitarySimulator(num_qubits=3).unitary

    for depth in prefix_depths:
        pattern = f"ppo_toffoli_prefix_identity_start0_target{depth}_seed*.json"
        # Each prefix has its own identity baseline.
        identity_fidelity = unitary_fidelity(identity, target_toffoli_prefix(depth))
        raw_fidelity = summarize_metric(pattern, "mean_final_fidelity")

        success.append(summarize_metric(pattern, "success_rate"))
        improvement.append((raw_fidelity[0] - identity_fidelity, raw_fidelity[1]))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(
        prefix_depths,
        [value for value, _ in improvement],
        yerr=[err for _, err in improvement],
        marker="o",
        linewidth=2,
        capsize=4,
        color=COLORS["orange"],
        label="fidelity above identity",
    )
    ax.errorbar(
        prefix_depths,
        [value for value, _ in success],
        yerr=[err for _, err in success],
        marker="s",
        linewidth=2,
        capsize=4,
        color=COLORS["blue"],
        label="success rate",
    )

    ax.axhline(0, color=COLORS["gray"], linestyle="-", linewidth=1)
    ax.axvline(3, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.text(
        3.15,
        -0.18,
        "only the\n3-gate prefix\nis reliable",
        color=COLORS["gray"],
        fontsize=9,
    )
    ax.set_title("Toffoli prefixes from identity: raw fidelity hides the identity plateau")
    ax.set_xlabel("target prefix depth")
    ax.set_ylabel("mean over 3 seeds")
    ax.set_ylim(-0.35, 1.05)
    ax.set_xticks(prefix_depths)
    ax.legend(loc="upper right")
    clean_axes(ax)
    save(fig, "figure_toffoli_prefix_from_identity.png")


def chunk_success(pattern: str) -> float:
    """Mean success rate for one curriculum chunk."""
    items = load_results(pattern)
    if not items:
        return np.nan
    return mean(item["success_rate"] for item in items)


def plot_curriculum_heatmap():
    """Plot which Toffoli curriculum chunks were solved."""
    columns = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
    rows = [
        (
            "Half chunks",
            [
                (7, "ppo_toffoli_prefix_continuation_start0_target7_max7_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start7_target15_max8_seed*.json"),
            ],
        ),
        (
            "Third chunks",
            [
                (5, "ppo_toffoli_prefix_continuation_start0_target5_max5_seed*.json"),
                (10, "ppo_toffoli_prefix_continuation_start5_target10_max5_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start10_target15_max5_seed*.json"),
            ],
        ),
        (
            "Quarter chunks\nexact depth",
            [
                (4, "ppo_toffoli_prefix_continuation_start0_target4_max4_seed*.json"),
                (8, "ppo_toffoli_prefix_continuation_start4_target8_max4_seed*.json"),
                (12, "ppo_toffoli_prefix_continuation_start8_target12_max4_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start12_target15_max3_seed*.json"),
            ],
        ),
        (
            "Quarter chunks\nmax depth 5",
            [
                (4, "ppo_toffoli_prefix_continuation_start0_target4_max5_seed*.json"),
                (8, "ppo_toffoli_prefix_continuation_start4_target8_max5_seed*.json"),
                (12, "ppo_toffoli_prefix_continuation_start8_target12_max5_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start12_target15_max5_seed*.json"),
            ],
        ),
        (
            "2-gate chunks\nexact depth",
            [
                (3, "ppo_toffoli_prefix_continuation_start0_target3_seed*.json"),
                (5, "ppo_toffoli_prefix_continuation_start3_target5_seed*.json"),
                (7, "ppo_toffoli_prefix_continuation_start5_target7_seed*.json"),
                (9, "ppo_toffoli_prefix_continuation_start7_target9_seed*.json"),
                (11, "ppo_toffoli_prefix_continuation_start9_target11_seed*.json"),
                (13, "ppo_toffoli_prefix_continuation_start11_target13_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start13_target15_seed*.json"),
            ],
        ),
        (
            "2-gate chunks\nmax depth 5",
            [
                (3, "ppo_toffoli_prefix_continuation_start0_target3_max5_seed*.json"),
                (5, "ppo_toffoli_prefix_continuation_start3_target5_max5_seed*.json"),
                (7, "ppo_toffoli_prefix_continuation_start5_target7_max5_seed*.json"),
                (9, "ppo_toffoli_prefix_continuation_start7_target9_max5_seed*.json"),
                (11, "ppo_toffoli_prefix_continuation_start9_target11_max5_seed*.json"),
                (13, "ppo_toffoli_prefix_continuation_start11_target13_max5_seed*.json"),
                (15, "ppo_toffoli_prefix_continuation_start13_target15_max5_seed*.json"),
            ],
        ),
    ]

    data = np.full((len(rows), len(columns)), np.nan)
    for row_index, (_, chunks) in enumerate(rows):
        for target_depth, pattern in chunks:
            col_index = columns.index(target_depth)
            data[row_index, col_index] = chunk_success(pattern)

    cmap = plt.cm.YlGn.copy()
    cmap.set_bad("#f0f1f3")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title("Toffoli curriculum: chunk size controls success")
    ax.set_xlabel("target prefix reached")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.grid(False)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if np.isnan(value):
                continue
            color = "white" if value > 0.7 else COLORS["dark"]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("success rate over 3 seeds")
    save(fig, "figure_toffoli_curriculum_heatmap.png")


def main():
    """Regenerate all report figures."""
    setup_style()
    plot_ghz_action_space()
    plot_toffoli_from_identity()
    plot_curriculum_heatmap()


if __name__ == "__main__":
    main()
