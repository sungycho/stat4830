"""Generate summary plots/tables from completed benchmark sweeps.

Assumes sweeps have already been run via src/run_sweep.py.

Usage:
  uv run python src/visualize_all.py
  uv run python src/visualize_all.py --sweeps full_comparison lqr_scaling
  uv run python src/visualize_all.py --results-root results --out-dir results/figures
  uv run python src/visualize_all.py --sweeps ars_variants alpha_sensitivity N_sensitivity
  uv run python src/visualize_all.py --sweeps eval100 --eval100-tasks swimmer hopper halfcheetah walker2d ant humanoid
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import (
    compute_seed_stats,
    compute_percentile_stats,
    load_runs_from_dir,
    plot_learning_curves,
    plot_scaling_comparison,
    summary_table,
    threshold_table,
    max_reward_table,
)


# ---------------------------- Config knobs ----------------------------
# Update these if your sweep names or locations differ.
RESULTS_ROOT_DEFAULT = "results"
FULL_COMPARISON_SWEEP = "full_comparison"
LQR_SCALING_SWEEP = "lqr_scaling"
SIGMA_SWEEP = "sigma_sensitivity"
ARS_VARIANTS_SWEEP = "ars_variants"
ALPHA_SWEEP = "alpha_sensitivity"
N_SWEEP = "N_sensitivity"
OUT_DIR_DEFAULT = "results/figures"

# Default thresholds per task (for threshold table)
TASK_THRESHOLDS = {
    "lqr": -50.0,
    "pendulum": -200.0,
    # MuJoCo thresholds from Mania et al. (Table 1)
    "swimmer": 325.0,
    "hopper": 3120.0,
    "halfcheetah": 3430.0,
    "walker2d": 4390.0,
    "ant": 3580.0,
    "humanoid": 6000.0,
}


METHOD_LABELS = {
    "ars": "ARS",
    "vanilla_es": "Vanilla ES",
    "reinforce": "REINFORCE",
}


def _safe_load_sweep(results_root: Path, sweep_name: str) -> list[dict]:
    sweep_dir = results_root / sweep_name
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Missing sweep directory: {sweep_dir}")
    runs = load_runs_from_dir(sweep_dir)
    if not runs:
        raise ValueError(f"No run JSON files found in: {sweep_dir}")
    return runs


def _group_by_method(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        method = run["config"]["method"]["name"]
        grouped.setdefault(method, []).append(run)
    return grouped


def _group_by_env_then_method(runs: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for run in runs:
        env_name = run["config"]["env"]["name"]
        method = run["config"]["method"]["name"]
        grouped.setdefault(env_name, {}).setdefault(method, []).append(run)
    return grouped


def _write_table(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def _write_sigma_csv(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "final_mean", "final_std", "final_median"])
        writer.writerows(rows)


def plot_percentile_curves(
    runs: list[dict],
    percentile_bands: list[tuple[float, float]] | None = None,
    title: str = "Learning Curves",
    save_path: str | Path | None = None,
) -> None:
    """Figure 1-style percentile band plot across seeds.

    Parameters
    ----------
    runs:
        List of per-seed run dicts.
    percentile_bands:
        List of (lo_pct, hi_pct) pairs. Defaults to [(0,10),(10,20),(20,100)].
    title:
        Figure title.
    save_path:
        Optional output path for the PNG.
    """
    if percentile_bands is None:
        percentile_bands = [(0, 10), (10, 20), (20, 100)]

    stats = compute_percentile_stats(runs, percentile_bands)
    episodes = stats["episodes"]
    median = stats["median"]

    alphas_band = [0.5, 0.35, 0.2]  # more opaque for tighter bands
    colors_band = ["steelblue", "cornflowerblue", "lightsteelblue"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, band in enumerate(stats["bands"]):
        alpha = alphas_band[i] if i < len(alphas_band) else 0.15
        color = colors_band[i] if i < len(colors_band) else "steelblue"
        label = f"{band['lo_pct']}–{band['hi_pct']}th pct"
        ax.fill_between(
            episodes,
            band["lo"],
            band["hi"],
            alpha=alpha,
            color=color,
            label=label,
        )

    ax.plot(episodes, median, color="steelblue", linestyle="--", linewidth=1.5, label="Median")
    ax.set_xlabel("Cumulative Episodes")
    ax.set_ylabel("Eval Return")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_full_comparison_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    runs = _safe_load_sweep(results_root, sweep_name)
    by_env = _group_by_env_then_method(runs)

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    for env_name, method_runs in by_env.items():
        method_stats: dict[str, dict] = {}
        for method_name, mruns in method_runs.items():
            label = METHOD_LABELS.get(method_name, method_name)
            method_stats[label] = compute_seed_stats(mruns)

        fig_path = sweep_out / f"full_comparison_{env_name}_curves.png"
        txt_path = sweep_out / f"full_comparison_{env_name}_summary.txt"

        plot_learning_curves(
            method_stats,
            title=f"Full Comparison ({env_name})",
            save_path=fig_path,
        )
        plt.close("all")
        _write_table(txt_path, summary_table(method_stats))


def make_lqr_scaling_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    runs = _safe_load_sweep(results_root, sweep_name)
    by_method: dict[str, dict[int, list[dict]]] = {}

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    for run in runs:
        method = run["config"]["method"]["name"]
        dim = int(run["config"]["env"]["state_dim"])
        by_method.setdefault(method, {}).setdefault(dim, []).append(run)

    sweep_results: dict[str, dict[int, dict]] = {}
    for method_name, dim_runs in by_method.items():
        label = METHOD_LABELS.get(method_name, method_name)
        sweep_results[label] = {}
        for dim, runs_for_dim in dim_runs.items():
            sweep_results[label][dim] = compute_seed_stats(runs_for_dim)

    fig_path = sweep_out / "lqr_scaling_final_return_vs_dimension.png"
    plot_scaling_comparison(sweep_results, save_path=fig_path)
    plt.close("all")


def make_sigma_sensitivity_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    runs = _safe_load_sweep(results_root, sweep_name)
    by_sigma: dict[float, list[dict]] = {}

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    for run in runs:
        sigma = float(run["config"]["method"]["sigma"])
        by_sigma.setdefault(sigma, []).append(run)

    sigmas = sorted(by_sigma.keys())
    final_means = []
    final_stds = []
    final_medians = []
    rows = []

    for sigma in sigmas:
        stats = compute_seed_stats(by_sigma[sigma])
        mean_val = float(stats["mean"][-1])
        std_val = float(stats["std"][-1])
        median_val = float(stats["median"][-1])
        final_means.append(mean_val)
        final_stds.append(std_val)
        final_medians.append(median_val)
        rows.append((sigma, mean_val, std_val, median_val))

    fig_path = sweep_out / "sigma_sensitivity_ars.png"
    plt.figure(figsize=(8, 5))
    plt.errorbar(sigmas, final_means, yerr=final_stds, marker="o", capsize=4)
    plt.xscale("log")
    plt.xlabel("Sigma")
    plt.ylabel("Final Eval Return")
    plt.title("ARS Sigma Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    csv_path = sweep_out / "sigma_sensitivity_ars_summary.csv"
    _write_sigma_csv(csv_path, rows)


def _group_by_variant(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by variant_label (falls back to method name)."""
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        label = run["config"]["method"].get("variant_label") or run["config"]["method"]["name"]
        grouped.setdefault(label, []).append(run)
    return grouped


def make_ars_variants_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    """Plot all ARS variants on one figure per env; produce threshold/max tables."""
    runs = _safe_load_sweep(results_root, sweep_name)
    by_env = _group_by_env_then_method(runs)

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    for env_name, method_runs in by_env.items():
        # Re-group by variant_label instead of method name
        variant_stats: dict[str, dict] = {}
        for mruns in method_runs.values():
            by_variant = _group_by_variant(mruns)
            for label, vruns in by_variant.items():
                variant_stats[label] = compute_seed_stats(vruns)

        fig_path = sweep_out / f"ars_variants_{env_name}_curves.png"
        txt_path = sweep_out / f"ars_variants_{env_name}_summary.txt"
        thr_path = sweep_out / f"ars_variants_{env_name}_threshold_table.txt"
        max_path = sweep_out / f"ars_variants_{env_name}_max_reward_table.txt"

        plot_learning_curves(
            variant_stats,
            title=f"ARS Variants ({env_name})",
            save_path=fig_path,
        )
        plt.close("all")
        _write_table(txt_path, summary_table(variant_stats))

        # Build runs_by_method for the table functions
        runs_by_variant: dict[str, list[dict]] = {}
        for mruns in method_runs.values():
            by_variant = _group_by_variant(mruns)
            for label, vruns in by_variant.items():
                runs_by_variant.setdefault(label, []).extend(vruns)

        thr = TASK_THRESHOLDS.get(env_name, 0.0)
        _write_table(thr_path, threshold_table(runs_by_variant, thr))
        _write_table(max_path, max_reward_table(runs_by_variant))


def _make_sensitivity_plot(
    by_param: dict,
    param_name: str,
    xlabel: str,
    title: str,
    save_path: Path,
    log_x: bool = True,
) -> None:
    param_vals = sorted(by_param.keys())
    final_means = []
    final_stds = []
    for pval in param_vals:
        stats = compute_seed_stats(by_param[pval])
        final_means.append(float(stats["mean"][-1]))
        final_stds.append(float(stats["std"][-1]))

    plt.figure(figsize=(8, 5))
    plt.errorbar(param_vals, final_means, yerr=final_stds, marker="o", capsize=4)
    if log_x:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Final Eval Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def make_alpha_sensitivity_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    """Plot final return vs learning rate α."""
    runs = _safe_load_sweep(results_root, sweep_name)
    by_alpha: dict[float, list[dict]] = {}
    for run in runs:
        alpha = float(run["config"]["method"]["lr"])
        by_alpha.setdefault(alpha, []).append(run)

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    _make_sensitivity_plot(
        by_alpha,
        param_name="alpha",
        xlabel="Learning Rate α",
        title="ARS Alpha Sensitivity (V2-t, LQR)",
        save_path=sweep_out / "alpha_sensitivity_ars.png",
        log_x=True,
    )


def make_N_sensitivity_outputs(results_root: Path, out_dir: Path, sweep_name: str) -> None:
    """Plot final return vs number of directions N."""
    runs = _safe_load_sweep(results_root, sweep_name)
    by_N: dict[int, list[dict]] = {}
    for run in runs:
        N = int(run["config"]["method"]["N"])
        by_N.setdefault(N, []).append(run)

    sweep_out = out_dir / sweep_name
    sweep_out.mkdir(parents=True, exist_ok=True)

    _make_sensitivity_plot(
        by_N,
        param_name="N",
        xlabel="Number of Directions N",
        title="ARS N Sensitivity (V2-t, LQR)",
        save_path=sweep_out / "N_sensitivity_ars.png",
        log_x=True,
    )


def make_eval100_outputs(
    results_root: Path,
    out_dir: Path,
    tasks: list[str],
    eval_subdir: str = "eval100",
) -> None:
    """Figure 1-style percentile plots for 100-seed evaluation results."""
    for task in tasks:
        task_dir = results_root / eval_subdir / task
        if not task_dir.exists():
            print(f"  eval100 dir not found: {task_dir}, skipping.")
            continue

        sweep_out = out_dir / eval_subdir / task
        sweep_out.mkdir(parents=True, exist_ok=True)

        # Expected layout: results/eval100/<task>/<variant>/*.json
        by_variant: dict[str, list[dict]] = {}
        for variant_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            vruns = load_runs_from_dir(variant_dir)
            if vruns:
                by_variant[variant_dir.name] = vruns

        for variant, vruns in by_variant.items():
            fig_path = sweep_out / f"eval100_{task}_{variant}_percentile.png"
            plot_percentile_curves(
                vruns,
                title=f"{variant} on {task} (n={len(vruns)} seeds)",
                save_path=fig_path,
            )
            print(f"  Saved: {fig_path}")


_ALL_SWEEPS = [
    "full_comparison",
    "lqr_scaling",
    "sigma_sensitivity",
    "ars_variants",
    "alpha_sensitivity",
    "N_sensitivity",
    "eval100",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots/tables from benchmark sweeps")
    parser.add_argument("--results-root", type=str, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=_ALL_SWEEPS,
        default=["full_comparison", "lqr_scaling", "sigma_sensitivity"],
        help="Which sweep visualizations to generate",
    )
    parser.add_argument("--full-sweep-name", type=str, default=FULL_COMPARISON_SWEEP)
    parser.add_argument("--lqr-sweep-name", type=str, default=LQR_SCALING_SWEEP)
    parser.add_argument("--sigma-sweep-name", type=str, default=SIGMA_SWEEP)
    parser.add_argument("--ars-variants-sweep-name", type=str, default=ARS_VARIANTS_SWEEP)
    parser.add_argument("--alpha-sweep-name", type=str, default=ALPHA_SWEEP)
    parser.add_argument("--N-sweep-name", type=str, default=N_SWEEP)
    parser.add_argument(
        "--eval100-tasks",
        nargs="+",
        default=["swimmer", "hopper", "halfcheetah", "walker2d", "ant", "humanoid"],
        help="Tasks to visualize for eval100 phase (used when 'eval100' is in --sweeps)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "full_comparison" in args.sweeps:
        make_full_comparison_outputs(results_root, out_dir, args.full_sweep_name)
    if "lqr_scaling" in args.sweeps:
        make_lqr_scaling_outputs(results_root, out_dir, args.lqr_sweep_name)
    if "sigma_sensitivity" in args.sweeps:
        make_sigma_sensitivity_outputs(results_root, out_dir, args.sigma_sweep_name)
    if "ars_variants" in args.sweeps:
        make_ars_variants_outputs(results_root, out_dir, args.ars_variants_sweep_name)
    if "alpha_sensitivity" in args.sweeps:
        make_alpha_sensitivity_outputs(results_root, out_dir, args.alpha_sweep_name)
    if "N_sensitivity" in args.sweeps:
        make_N_sensitivity_outputs(results_root, out_dir, args.N_sweep_name)
    if "eval100" in args.sweeps:
        make_eval100_outputs(results_root, out_dir, tasks=args.eval100_tasks)

    print(f"Saved visualizations to: {out_dir}")
    print(f"Selected sweeps: {args.sweeps}")
    print("Generated per-sweep subfolders under out-dir.")


if __name__ == "__main__":
    main()
