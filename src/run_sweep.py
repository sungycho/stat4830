"""CLI entry point for running benchmark sweeps.

Usage:
    uv run python src/run_sweep.py --sweep lqr_scaling --budget 3200
    uv run python src/run_sweep.py --sweep full_comparison --budget 3200
    uv run python src/run_sweep.py --sweep sigma_sensitivity --budget 3200
    uv run python src/run_sweep.py --sweep ars_variants --budget 1600
    uv run python src/run_sweep.py --sweep alpha_sensitivity --budget 3200
    uv run python src/run_sweep.py --sweep N_sensitivity --budget 3200
    uv run python src/run_sweep.py --sweep mujoco_comparison --budget 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig, EnvConfig, MethodConfig, SweepConfig
from src.runner import run_sweep, run_multi_seed


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


def make_ars_variants(budget: int) -> SweepConfig:
    """Compare BRS, V1, V1-t, V2, V2-t on LQR and Pendulum.

    Variant definitions (paper Table A.2):
      BRS:   reward_norm=False, use_state_norm=False, b=N
      V1:    reward_norm=True,  use_state_norm=False, b=N
      V1-t:  reward_norm=True,  use_state_norm=False, b<N
      V2:    reward_norm=True,  use_state_norm=True,  b=N
      V2-t:  reward_norm=True,  use_state_norm=True,  b<N
    """
    N = 16
    b = 8  # top-b < N for all non-BRS variants

    variant_defs = [
        # (label, reward_norm, use_state_norm, b_val)
        ("BRS",  False, False, N),
        ("V1",   True,  False, N),
        ("V1-t", True,  False, b),
        ("V2",   True,  True,  N),
        ("V2-t", True,  True,  b),
    ]

    variants = []
    for env_name in ["lqr", "pendulum"]:
        for label, rew_norm, state_norm, b_val in variant_defs:
            v: dict = {
                "env.name": env_name,
                "method.name": "ars",
                "method.N": N,
                "method.b": b_val,
                "method.reward_norm": rew_norm,
                "method.use_state_norm": state_norm,
                "method.variant_label": label,
            }
            if env_name == "pendulum":
                v["env.state_dim"] = 3
                v["env.action_dim"] = 1
            variants.append(v)

    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="ars_variants")


def make_alpha_sensitivity(budget: int) -> SweepConfig:
    """Sweep learning rate α for ARS V2-t on LQR (Appendix A.3)."""
    variants = []
    for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        variants.append({
            "method.name": "ars",
            "method.lr": alpha,
            "method.reward_norm": True,
            "method.use_state_norm": True,
            "method.variant_label": "V2-t",
        })
    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="alpha_sensitivity")


def make_N_sensitivity(budget: int) -> SweepConfig:
    """Sweep N (directions) for ARS V2-t on LQR (Appendix A.3).

    b is always set to N // 2.
    """
    variants = []
    for N in [4, 8, 16, 32, 64]:
        variants.append({
            "method.name": "ars",
            "method.N": N,
            "method.b": N // 2,
            "method.reward_norm": True,
            "method.use_state_norm": True,
            "method.variant_label": "V2-t",
        })
    base = _base_config(budget)
    return SweepConfig(base=base, variants=variants, sweep_name="N_sensitivity")


def make_mujoco_comparison(budget: int) -> SweepConfig:
    """ARS V2-t on all 6 MuJoCo locomotion tasks from the paper."""
    # Survival bonus tasks: hopper, walker2d, ant, humanoid
    tasks = ["swimmer", "hopper", "halfcheetah", "walker2d", "ant", "humanoid"]
    survival_bonus_tasks = {"hopper", "walker2d", "ant", "humanoid"}

    variants = []
    for task in tasks:
        v: dict = {
            "env.name": task,
            "method.name": "ars",
            "method.reward_norm": True,
            "method.use_state_norm": True,
            "method.variant_label": "V2-t",
        }
        if task in survival_bonus_tasks:
            v["env.remove_survival_bonus"] = True
        variants.append(v)

    base = _base_config(budget)
    base.max_steps = 1000
    return SweepConfig(base=base, variants=variants, sweep_name="mujoco_comparison")


SWEEPS = {
    "lqr_scaling": make_lqr_scaling,
    "full_comparison": make_full_comparison,
    "sigma_sensitivity": make_sigma_sensitivity,
    "ars_variants": make_ars_variants,
    "alpha_sensitivity": make_alpha_sensitivity,
    "N_sensitivity": make_N_sensitivity,
    "mujoco_comparison": make_mujoco_comparison,
}


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweeps")
    parser.add_argument("--sweep", choices=list(SWEEPS.keys()), required=True)
    parser.add_argument("--budget", type=int, default=3200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--use-ray",
        action="store_true",
        default=False,
        help="Enable Ray-parallel rollouts for ARS/BRS sweeps.",
    )
    parser.add_argument(
        "--noise-table-size",
        type=int,
        default=0,
        help="Max noise table entries for Ray ARS (0 = auto, up to 250 M).",
    )
    args = parser.parse_args()

    sweep_fn = SWEEPS[args.sweep]
    sweep = sweep_fn(args.budget)
    sweep.base.seeds = args.seeds
    sweep.base.results_dir = args.results_dir
    if args.use_ray:
        sweep.base.method.use_ray = True
    if args.noise_table_size > 0:
        sweep.base.method.noise_table_size = args.noise_table_size

    results_dir = Path(args.results_dir) / sweep.sweep_name
    print(f"Running sweep '{args.sweep}' with budget={args.budget}, seeds={args.seeds}")
    print(f"Results dir: {results_dir}")
    print(f"Total variant runs: {len(sweep.variants) * len(sweep.base.seeds)}")

    paths = run_sweep(sweep, results_dir)
    print(f"\nCompleted. Wrote {len(paths)} result files to {results_dir}")


if __name__ == "__main__":
    main()
