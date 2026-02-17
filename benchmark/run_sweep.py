"""CLI entry point for running benchmark sweeps.

Usage:
    python -m benchmark.run_sweep --sweep lqr_scaling --budget 3200
    python -m benchmark.run_sweep --sweep full_comparison --budget 3200
    python -m benchmark.run_sweep --sweep sigma_sensitivity --budget 3200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.config import ExperimentConfig, EnvConfig, MethodConfig, SweepConfig
from benchmark.runner import run_sweep, run_multi_seed


def _base_config(budget: int, seeds: list[int] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        total_episode_budget=budget,
        seeds=seeds or [0, 1, 2],
    )


def make_lqr_scaling(budget: int) -> SweepConfig:
    """dims [4, 16, 64, 256] x 3 methods."""
    variants = []
    for dim in [4, 16, 64, 256]:
        a_dim = max(1, dim // 4)
        for method in ["ars", "vanilla_es", "reinforce"]:
            variants.append({
                "env.state_dim": dim,
                "env.action_dim": a_dim,
                "method.name": method,
            })
    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="lqr_scaling")


def make_full_comparison(budget: int) -> SweepConfig:
    """3 methods x 2 envs."""
    variants = []
    for method in ["ars", "vanilla_es", "reinforce"]:
        variants.append({"method.name": method, "env.name": "lqr"})
        variants.append({
            "method.name": method,
            "env.name": "pendulum",
            "env.state_dim": 3,
            "env.action_dim": 1,
        })
    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="full_comparison")


def make_sigma_sensitivity(budget: int) -> SweepConfig:
    """Sweep sigma in [0.005, 0.01, 0.03, 0.1, 0.3] for ARS on LQR."""
    variants = []
    for sigma in [0.005, 0.01, 0.03, 0.1, 0.3]:
        variants.append({"method.name": "ars", "method.sigma": sigma})
    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="sigma_sensitivity")


SWEEPS = {
    "lqr_scaling": make_lqr_scaling,
    "full_comparison": make_full_comparison,
    "sigma_sensitivity": make_sigma_sensitivity,
}


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweeps")
    parser.add_argument("--sweep", choices=list(SWEEPS.keys()), required=True)
    parser.add_argument("--budget", type=int, default=3200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    sweep_fn = SWEEPS[args.sweep]
    sweep = sweep_fn(args.budget)
    sweep.base.seeds = args.seeds
    sweep.base.results_dir = args.results_dir

    results_dir = Path(args.results_dir) / sweep.sweep_name
    print(f"Running sweep '{args.sweep}' with budget={args.budget}, seeds={args.seeds}")
    print(f"Results dir: {results_dir}")
    print(f"Total variant runs: {len(sweep.variants) * len(sweep.base.seeds)}")

    paths = run_sweep(sweep, results_dir)
    print(f"\nCompleted. Wrote {len(paths)} result files to {results_dir}")


if __name__ == "__main__":
    main()
