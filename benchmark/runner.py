"""Orchestration: run experiments, save JSON results."""

from __future__ import annotations

import json
import time
from pathlib import Path

from benchmark.config import ExperimentConfig, SweepConfig
from benchmark.envs.lqr import LQREnv
from benchmark.envs.pendulum import PendulumEnv
from benchmark.methods.ars import run_ars
from benchmark.methods.vanilla_es import run_vanilla_es
from benchmark.methods.reinforce import run_reinforce


def build_env(config: ExperimentConfig):
    ec = config.env
    if ec.name == "lqr":
        return LQREnv(
            state_dim=ec.state_dim,
            action_dim=ec.action_dim,
            horizon=ec.horizon,
            noise_std=ec.noise_std,
            system_seed=ec.system_seed,
        )
    elif ec.name == "pendulum":
        return PendulumEnv(
            obs_noise_std=ec.noise_std,
            max_steps=config.max_steps,
        )
    raise ValueError(f"Unknown env: {ec.name}")


def build_method(config: ExperimentConfig):
    name = config.method.name
    if name == "ars":
        return run_ars
    elif name == "vanilla_es":
        return run_vanilla_es
    elif name == "reinforce":
        return run_reinforce
    raise ValueError(f"Unknown method: {name}")


def run_single(config: ExperimentConfig, seed: int, results_dir: str | Path) -> Path:
    """Run one (config, seed) experiment and save results as JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(config)
    method_fn = build_method(config)

    t0 = time.time()
    iteration_results = method_fn(config, env, seed)
    elapsed = time.time() - t0

    env.close()

    tag = config.run_tag or f"{config.method.name}_{config.env.name}"
    out_path = results_dir / f"{tag}_seed{seed}.json"

    data = {
        "config": config.to_dict(),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "iterations": [r.to_dict() for r in iteration_results],
    }
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def run_multi_seed(config: ExperimentConfig, results_dir: str | Path) -> list[Path]:
    """Run all seeds for a given config."""
    return [run_single(config, seed, results_dir) for seed in config.seeds]


def run_sweep(sweep: SweepConfig, results_dir: str | Path) -> list[Path]:
    """Run all variants in a sweep."""
    all_paths = []
    for i, overrides in enumerate(sweep.variants):
        cfg = sweep.apply_overrides(overrides)
        tag_parts = [f"{k.split('.')[-1]}={v}" for k, v in overrides.items()]
        cfg.run_tag = f"{sweep.sweep_name}_v{i}_" + "_".join(tag_parts)
        paths = run_multi_seed(cfg, results_dir)
        all_paths.extend(paths)
    return all_paths
