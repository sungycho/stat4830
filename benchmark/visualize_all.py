"""Generate summary plots/tables from completed benchmark sweeps.

Assumes these sweeps have already been run:
  - full_comparison
  - lqr_scaling
  - sigma_sensitivity

Usage:
  python -m benchmark.visualize_all
  python -m benchmark.visualize_all --sweeps full_comparison lqr_scaling
  python -m benchmark.visualize_all --results-root benchmark_results --out-dir benchmark_results/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.analysis import (
    compute_seed_stats,
    load_runs_from_dir,
    plot_learning_curves,
    plot_scaling_comparison,
    summary_table,
)


# ---------------------------- Config knobs ----------------------------
# Update these if your sweep names or locations differ.
RESULTS_ROOT_DEFAULT = "benchmark_results"
FULL_COMPARISON_SWEEP = "full_comparison"
LQR_SCALING_SWEEP = "lqr_scaling"
SIGMA_SWEEP = "sigma_sensitivity"
OUT_DIR_DEFAULT = "benchmark_results/figures"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots/tables from benchmark sweeps")
    parser.add_argument("--results-root", type=str, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=["full_comparison", "lqr_scaling", "sigma_sensitivity"],
        default=["full_comparison", "lqr_scaling", "sigma_sensitivity"],
        help="Which sweep visualizations to generate",
    )
    parser.add_argument("--full-sweep-name", type=str, default=FULL_COMPARISON_SWEEP)
    parser.add_argument("--lqr-sweep-name", type=str, default=LQR_SCALING_SWEEP)
    parser.add_argument("--sigma-sweep-name", type=str, default=SIGMA_SWEEP)
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

    print(f"Saved visualizations to: {out_dir}")
    print(f"Selected sweeps: {args.sweeps}")
    print("Generated per-sweep subfolders under out-dir.")


if __name__ == "__main__":
    main()
