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


# ---------------------------------------------------------------------------
# Paper-specific analysis: episodes-to-threshold, percentile stats, tables
# ---------------------------------------------------------------------------

def episodes_to_threshold(run: dict, threshold: float) -> int | None:
    """Return the first episodes_consumed where eval_return >= threshold.

    Scans iterations in order. Returns None if the threshold is never reached.
    """
    for it in run["iterations"]:
        if it["eval_return"] is not None and it["eval_return"] >= threshold:
            return it["episodes_consumed"]
    return None


def compute_percentile_stats(
    runs: list[dict],
    percentile_pairs: list[tuple[float, float]] | None = None,
) -> dict:
    """Compute per-band percentile arrays across seeds.

    Parameters
    ----------
    runs:
        List of run dicts (one per seed).
    percentile_pairs:
        List of (lo_pct, hi_pct) pairs defining shaded bands.
        Defaults to [(0, 10), (10, 20), (20, 100)] (Figure 1 style).

    Returns
    -------
    dict with keys:
      "episodes": list of cumulative episode counts
      "median":   list of median eval returns
      "bands":    list of dicts, each with lo_pct, hi_pct, lo (array), hi (array)
    """
    if percentile_pairs is None:
        percentile_pairs = [(0, 10), (10, 20), (20, 100)]

    curves = [extract_eval_curve(r) for r in runs]
    episodes = curves[0][0]
    all_returns = np.array([c[1] for c in curves])  # (n_seeds, n_evals)

    bands = []
    for lo_pct, hi_pct in percentile_pairs:
        bands.append({
            "lo_pct": lo_pct,
            "hi_pct": hi_pct,
            "lo": np.percentile(all_returns, lo_pct, axis=0).tolist(),
            "hi": np.percentile(all_returns, hi_pct, axis=0).tolist(),
        })

    return {
        "episodes": episodes,
        "median": np.median(all_returns, axis=0).tolist(),
        "bands": bands,
    }


def select_best_config(
    grid_results: dict[str, list[dict]],
    threshold: float,
) -> str:
    """Select the config key that minimizes mean episodes-to-threshold.

    Parameters
    ----------
    grid_results:
        Mapping from config_key (str) → list of run dicts (one per seed).
    threshold:
        Reward threshold defining "solved".

    Returns
    -------
    The config_key with the lowest mean episodes-to-threshold.
    If no config ever reaches the threshold for any seed, returns the key
    with the highest mean final eval return.
    """
    best_key: str | None = None
    best_score = float("inf")

    for key, runs in grid_results.items():
        etts = [episodes_to_threshold(r, threshold) for r in runs]
        # Replace None (never reached) with budget (worst case)
        budget = max(
            r["iterations"][-1]["episodes_consumed"] for r in runs
        )
        numeric = [e if e is not None else budget for e in etts]
        mean_score = float(np.mean(numeric))
        if mean_score < best_score:
            best_score = mean_score
            best_key = key

    if best_key is None:
        raise ValueError("grid_results is empty")
    return best_key


def threshold_table(
    runs_by_method: dict[str, list[dict]],
    thresholds: dict[str, float] | float,
) -> str:
    """Format a Table 1-style threshold table.

    Parameters
    ----------
    runs_by_method:
        {method_label: list_of_runs}
    thresholds:
        Either a single float threshold, or a dict mapping method → threshold.
        If a single float, the same threshold is used for all methods.

    Returns
    -------
    Formatted text table: method | mean episodes | median episodes | % reached
    """
    lines = [
        f"{'Method':<15} {'Mean Eps':>10} {'Median Eps':>12} {'% Reached':>10}"
    ]
    lines.append("-" * 50)

    for method, runs in runs_by_method.items():
        if isinstance(thresholds, dict):
            thr = thresholds.get(method, thresholds.get("default", 0.0))
        else:
            thr = float(thresholds)

        budget = max(r["iterations"][-1]["episodes_consumed"] for r in runs)
        etts = [episodes_to_threshold(r, thr) for r in runs]
        reached = [e for e in etts if e is not None]
        pct = 100.0 * len(reached) / len(etts) if etts else 0.0
        numeric = [e if e is not None else budget for e in etts]
        mean_eps = float(np.mean(numeric))
        median_eps = float(np.median(numeric))
        lines.append(
            f"{method:<15} {mean_eps:>10.0f} {median_eps:>12.0f} {pct:>9.1f}%"
        )

    return "\n".join(lines)


def max_reward_table(runs_by_method: dict[str, list[dict]]) -> str:
    """Format a Table 2-style maximum reward table.

    For each method, reports max eval_return across all seeds and iterations.
    Also shows mean and std of per-seed max returns.
    """
    lines = [
        f"{'Method':<15} {'Max Return':>12} {'Mean Max':>10} {'Std Max':>10}"
    ]
    lines.append("-" * 50)

    for method, runs in runs_by_method.items():
        per_seed_max = []
        for run in runs:
            evals = [it["eval_return"] for it in run["iterations"]
                     if it["eval_return"] is not None]
            per_seed_max.append(max(evals) if evals else float("nan"))
        overall_max = max(per_seed_max)
        mean_max = float(np.nanmean(per_seed_max))
        std_max = float(np.nanstd(per_seed_max))
        lines.append(
            f"{method:<15} {overall_max:>12.2f} {mean_max:>10.2f} {std_max:>10.2f}"
        )

    return "\n".join(lines)
