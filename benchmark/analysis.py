"""Offline analysis: load results, compute stats, plot curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_runs_from_dir(dir_path: str | Path) -> list[dict]:
    d = Path(dir_path)
    return [load_run(p) for p in sorted(d.glob("*.json"))]


def extract_eval_curve(run: dict) -> tuple[list[int], list[float]]:
    """Extract (cumulative_episodes, eval_returns) where eval was recorded."""
    episodes = []
    returns = []
    for it in run["iterations"]:
        if it["eval_return"] is not None:
            episodes.append(it["episodes_consumed"])
            returns.append(it["eval_return"])
    return episodes, returns


def compute_seed_stats(runs: list[dict]) -> dict:
    """Compute stats across seeds. Assumes all runs share eval schedule."""
    curves = [extract_eval_curve(r) for r in runs]
    # Use first run's episode axis
    episodes = curves[0][0]
    all_returns = np.array([c[1] for c in curves])  # (n_seeds, n_evals)

    return {
        "episodes": episodes,
        "mean": np.mean(all_returns, axis=0).tolist(),
        "std": np.std(all_returns, axis=0).tolist(),
        "median": np.median(all_returns, axis=0).tolist(),
        "p25": np.percentile(all_returns, 25, axis=0).tolist(),
        "p75": np.percentile(all_returns, 75, axis=0).tolist(),
    }


def plot_learning_curves(
    method_stats: dict[str, dict],
    title: str = "Learning Curves",
    save_path: str | Path | None = None,
):
    """Plot mean + shaded std bands per method.

    method_stats: {method_name: output of compute_seed_stats}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, stats in method_stats.items():
        eps = stats["episodes"]
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        ax.plot(eps, mean, label=name)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Cumulative Episodes")
    ax.set_ylabel("Eval Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scaling_comparison(
    sweep_results: dict[str, dict[int, dict]],
    save_path: str | Path | None = None,
):
    """Plot x=state_dim (log), y=final eval return per method.

    sweep_results: {method_name: {state_dim: seed_stats}}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, dim_stats in sweep_results.items():
        dims = sorted(dim_stats.keys())
        final_means = [dim_stats[d]["mean"][-1] for d in dims]
        final_stds = [dim_stats[d]["std"][-1] for d in dims]
        ax.errorbar(dims, final_means, yerr=final_stds, label=name, marker="o", capsize=4)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("State Dimension")
    ax.set_ylabel("Final Eval Return")
    ax.set_title("Scaling Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def summary_table(method_stats: dict[str, dict]) -> str:
    """Formatted text table of final eval stats per method."""
    lines = [f"{'Method':<15} {'Final Mean':>12} {'Final Std':>12} {'Final Median':>14}"]
    lines.append("-" * 55)
    for name, stats in method_stats.items():
        m = stats["mean"][-1]
        s = stats["std"][-1]
        med = stats["median"][-1]
        lines.append(f"{name:<15} {m:>12.2f} {s:>12.2f} {med:>14.2f}")
    return "\n".join(lines)
