"""Two-phase evaluation protocol matching Mania et al. NeurIPS 2018.

Phase 1 — Hyperparameter grid search (3 seeds per config):
  For each task × variant:
    For each (α, N, ν) in the grid:
      Run 3 fixed seeds
      Compute mean episodes-to-threshold
    Select best config (min mean episodes-to-threshold)
  Save best_configs.json

Phase 2 — 100-seed evaluation:
  For each task × best_variant_config:
    Run seeds 0..99
    Save all run JSONs to results/<sweep>/

CLI usage:
  uv run python src/sweep_protocol.py --phase grid \\
      --tasks swimmer hopper halfcheetah walker2d ant humanoid \\
      --variants V1 V1-t V2 V2-t \\
      --results-dir results

  uv run python src/sweep_protocol.py --phase eval100 \\
      --best-configs results/best_configs.json \\
      --results-dir results
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from src.config import ExperimentConfig, EnvConfig, MethodConfig
from src.runner import run_single
from src.analysis import episodes_to_threshold


# ---------------------------------------------------------------------------
# Default thresholds (reward where "solved" is declared) per task
# ---------------------------------------------------------------------------
TASK_THRESHOLDS: dict[str, float] = {
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

# ARS variant definitions aligned with paper Algorithm 1:
# (reward_norm, use_state_norm, use_top_b)
# - V1/V2 use all directions (b=N)
# - V1-t/V2-t use top-b (b<N)
VARIANT_DEFS: dict[str, tuple[bool, bool, bool]] = {
    "BRS":  (False, False, False),
    "V1":   (True,  False, False),
    "V1-t": (True,  False, True),
    "V2":   (True,  True,  False),
    "V2-t": (True,  True,  True),
}

# Task → max_steps for rollouts
TASK_MAX_STEPS: dict[str, int] = {
    "lqr": 200,
    "pendulum": 200,
    "swimmer": 1000,
    "hopper": 1000,
    "halfcheetah": 1000,
    "walker2d": 1000,
    "ant": 1000,
    "humanoid": 1000,
}

_MUJOCO_TASKS = {"swimmer", "hopper", "halfcheetah", "walker2d", "ant", "humanoid"}
_SURVIVAL_BONUS_TASKS = {"hopper", "walker2d", "ant", "humanoid"}


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _make_config(
    task: str,
    variant_label: str,
    alpha: float,
    N: int,
    sigma: float,
    budget: int,
    seeds: list[int],
    use_ray: bool = False,
    noise_table_size: int = 0,
) -> ExperimentConfig:
    reward_norm, use_state_norm, use_top_b = VARIANT_DEFS[variant_label]
    b = (N // 2) if use_top_b else N

    env_cfg = EnvConfig(name=task)
    if task == "pendulum":
        env_cfg.state_dim = 3
        env_cfg.action_dim = 1
    elif task in _MUJOCO_TASKS:
        env_cfg.remove_survival_bonus = task in _SURVIVAL_BONUS_TASKS

    method_cfg = MethodConfig(
        name="ars",
        lr=alpha,
        N=N,
        b=b,
        sigma=sigma,
        reward_norm=reward_norm,
        use_state_norm=use_state_norm,
        variant_label=variant_label,
        use_ray=use_ray,
        noise_table_size=noise_table_size,
    )

    return ExperimentConfig(
        env=env_cfg,
        method=method_cfg,
        total_episode_budget=budget,
        seeds=seeds,
        max_steps=TASK_MAX_STEPS.get(task, 200),
        eval_every_iters=10,
        # Paper reports policy performance averaged over 100 rollouts.
        eval_episodes=100,
    )


def _sample_fixed_seeds(n: int, pool_size: int, rng_seed: int) -> list[int]:
    """Deterministically sample n distinct seeds from [0, pool_size)."""
    import numpy as np
    if n > pool_size:
        raise ValueError(f"n={n} cannot exceed pool_size={pool_size}")
    rng = np.random.default_rng(rng_seed)
    return rng.choice(pool_size, size=n, replace=False).tolist()


# ---------------------------------------------------------------------------
# Phase 1: grid search
# ---------------------------------------------------------------------------

def run_grid_phase(
    tasks: list[str],
    variants: list[str],
    results_dir: Path,
    budget: int = 3200,
    seeds: list[int] | None = None,
    alphas: list[float] | None = None,
    N_values: list[int] | None = None,
    sigmas: list[float] | None = None,
    use_ray: bool = False,
    noise_table_size: int = 0,
) -> dict:
    """Run hyperparameter grid and return best config per (task, variant).

    Returns a dict suitable for saving as best_configs.json.
    """
    # Paper protocol: 3 fixed seeds sampled from [0, 1000).
    seeds = seeds or _sample_fixed_seeds(n=3, pool_size=1000, rng_seed=0)
    alphas = alphas or [0.01, 0.02, 0.05]
    N_values = N_values or [8, 16, 32]
    sigmas = sigmas or [0.01, 0.03, 0.1]

    grid_dir = results_dir / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    best_configs: dict[str, dict] = {}

    for task in tasks:
        threshold = TASK_THRESHOLDS.get(task, 0.0)
        best_configs[task] = {}

        for variant in variants:
            print(f"\n=== Grid search: task={task}, variant={variant} ===")
            best_key: str | None = None
            best_score: float = float("inf")
            config_scores: dict[str, float] = {}

            for alpha, N, sigma in product(alphas, N_values, sigmas):
                config = _make_config(
                    task=task,
                    variant_label=variant,
                    alpha=alpha,
                    N=N,
                    sigma=sigma,
                    budget=budget,
                    seeds=seeds,
                    use_ray=use_ray,
                    noise_table_size=noise_table_size,
                )
                key = f"alpha={alpha}_N={N}_sigma={sigma}"
                run_dir = grid_dir / task / variant / key
                run_dir.mkdir(parents=True, exist_ok=True)

                episode_counts: list[float] = []
                for seed in seeds:
                    config.run_tag = f"{task}_{variant}_{key}_seed{seed}"
                    try:
                        path = run_single(config, seed, run_dir)
                        run_data = json.loads(path.read_text())
                        etts = episodes_to_threshold(run_data, threshold)
                        if etts is None:
                            # Never reached threshold: penalize with full budget
                            episode_counts.append(float(budget))
                        else:
                            episode_counts.append(float(etts))
                        print(f"  {key} seed={seed}: episodes_to_threshold={etts}")
                    except Exception as e:
                        print(f"  {key} seed={seed}: ERROR {e}")
                        episode_counts.append(float(budget))

                mean_score = sum(episode_counts) / len(episode_counts)
                config_scores[key] = mean_score
                print(f"  → mean episodes-to-threshold: {mean_score:.1f}")

                if mean_score < best_score:
                    best_score = mean_score
                    best_key = key

            entry: dict = {
                "key": best_key,
                "mean_episodes_to_threshold": best_score,
                "all_scores": config_scores,
            }
            if best_key:
                entry.update(_parse_key(best_key))
            best_configs[task][variant] = entry
            print(f"  ★ Best for {task}/{variant}: {best_key} (score={best_score:.1f})")

    out_path = results_dir / "best_configs.json"
    out_path.write_text(json.dumps(best_configs, indent=2))
    print(f"\nSaved best_configs to: {out_path}")
    return best_configs


def _parse_key(key: str) -> dict:
    """Parse 'alpha=0.02_N=16_sigma=0.03' → {'alpha': 0.02, 'N': 16, 'sigma': 0.03}."""
    result: dict = {}
    for part in key.split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                result[k] = int(v)
            except ValueError:
                result[k] = float(v)
    return result


# ---------------------------------------------------------------------------
# Phase 2: 100-seed evaluation
# ---------------------------------------------------------------------------

def run_eval100_phase(
    best_configs_path: Path,
    results_dir: Path,
    budget: int = 3200,
    n_seeds: int = 100,
    seed_pool_size: int = 10000,
    seed_rng_seed: int = 0,
    use_ray: bool = False,
    noise_table_size: int = 0,
) -> list[Path]:
    """Run 100-seed evaluation using configs selected in Phase 1."""
    best_configs = json.loads(best_configs_path.read_text())

    eval_dir = results_dir / "eval100"
    eval_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []

    eval_seeds = _sample_fixed_seeds(n=n_seeds, pool_size=seed_pool_size, rng_seed=seed_rng_seed)

    for task, variant_map in best_configs.items():
        for variant, info in variant_map.items():
            alpha = float(info.get("alpha", 0.02))
            N = int(info.get("N", 16))
            sigma = float(info.get("sigma", 0.03))

            print(f"\n=== Eval100: task={task}, variant={variant} "
                  f"(α={alpha}, N={N}, σ={sigma}) ===")

            config = _make_config(
                task=task,
                variant_label=variant,
                alpha=alpha,
                N=N,
                sigma=sigma,
                budget=budget,
                seeds=eval_seeds,
                use_ray=use_ray,
                noise_table_size=noise_table_size,
            )

            run_dir = eval_dir / task / variant
            run_dir.mkdir(parents=True, exist_ok=True)

            for i, seed in enumerate(eval_seeds):
                config.run_tag = f"eval100_{task}_{variant}_seed{seed}"
                try:
                    path = run_single(config, seed, run_dir)
                    all_paths.append(path)
                    if i % 10 == 0:
                        print(f"  seed {i + 1}/{n_seeds} (id={seed}) done → {path.name}")
                except Exception as e:
                    print(f"  seed {seed}: ERROR {e}")

    print(f"\nEval100 complete. Wrote {len(all_paths)} files to {eval_dir}")
    return all_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase ARS evaluation protocol (Mania et al. 2018)"
    )
    parser.add_argument(
        "--phase",
        choices=["grid", "eval100"],
        required=True,
        help="grid: hyperparameter search (3 seeds). eval100: 100-seed evaluation.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["swimmer", "hopper", "halfcheetah", "walker2d", "ant", "humanoid"],
        help="Environments to evaluate on.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["V1", "V1-t", "V2", "V2-t"],
        choices=list(VARIANT_DEFS),
        help="ARS variants to evaluate.",
    )
    parser.add_argument(
        "--best-configs",
        type=str,
        default=None,
        help="Path to best_configs.json (required for --phase eval100).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory.",
    )
    parser.add_argument("--budget", type=int, default=3200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-seeds", type=int, default=100, help="Seeds for eval100 phase.")
    parser.add_argument(
        "--seed-pool-size",
        type=int,
        default=10000,
        help="Sample eval100 seeds from [0, seed-pool-size).",
    )
    parser.add_argument(
        "--seed-rng-seed",
        type=int,
        default=0,
        help="RNG seed used to sample fixed seed sets.",
    )
    parser.add_argument(
        "--use-ray",
        action="store_true",
        default=False,
        help="Enable Ray-parallel rollouts for ARS variants.",
    )
    parser.add_argument(
        "--noise-table-size",
        type=int,
        default=0,
        help="Max noise table entries for Ray ARS (0 = auto, up to 250 M).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "grid":
        run_grid_phase(
            tasks=args.tasks,
            variants=args.variants,
            results_dir=results_dir,
            budget=args.budget,
            seeds=args.seeds,
            use_ray=args.use_ray,
            noise_table_size=args.noise_table_size,
        )
    elif args.phase == "eval100":
        if args.best_configs is None:
            parser.error("--best-configs is required for --phase eval100")
        run_eval100_phase(
            best_configs_path=Path(args.best_configs),
            results_dir=results_dir,
            budget=args.budget,
            n_seeds=args.n_seeds,
            seed_pool_size=args.seed_pool_size,
            seed_rng_seed=args.seed_rng_seed,
            use_ray=args.use_ray,
            noise_table_size=args.noise_table_size,
        )


if __name__ == "__main__":
    main()
