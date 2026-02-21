# STAT 4830 Project: Reproducing Augmented Random Search (Mania et al., NeurIPS 2018)

This repository reproduces the core experimental findings of **"Simple random search of static linear policies is competitive for reinforcement learning"** (Mania, Guy, Recht, NeurIPS 2018) and benchmarks ARS against Vanilla ES and REINFORCE on continuous-control environments.

---

## Project Overview

The paper's central claim is that Augmented Random Search (ARS) — a simple derivative-free method using antithetic perturbations, reward normalization, and optional state normalization — matches or exceeds state-of-the-art deep RL on MuJoCo locomotion tasks. This project:

1. Implements all five ARS variants (BRS, V1, V1-t, V2, V2-t) from the paper
2. Benchmarks against Vanilla ES and REINFORCE under matched episode budgets
3. Runs the paper's two-phase evaluation protocol (hyperparameter grid search → multi-seed evaluation)
4. Conducts sensitivity analyses over σ, α, and N
5. Studies scaling behavior across LQR problem dimensions (4 → 256)

---

## Key Results (LQR + Pendulum, budget=3200)

| Method | LQR final return | Pendulum final return |
|--------|----------------:|-----------------------:|
| ARS V2-t | **−16.6** | **−1,162** |
| Vanilla ES | −179,104 | −1,734 |
| REINFORCE | −178,106 | −1,601 |

- ARS converges ~10,800× better than baselines on LQR
- Reward normalization alone (BRS → V1) accounts for a 14,600× improvement
- ARS degrades polynomially with dimension; ES/PG degrade exponentially (see `results/figures/lqr_scaling/`)

---

## Setup

```bash
# Python 3.13, managed with uv
uv sync
source .venv/bin/activate
```

To enable MuJoCo environments (optional):
```bash
uv add "gymnasium[mujoco]"
```

---

## Running Experiments

### Sweep runner (LQR, Pendulum, sensitivity analyses)

```bash
# Full method comparison (ARS vs Vanilla ES vs REINFORCE)
uv run python src/run_sweep.py --sweep full_comparison --budget 3200

# ARS variant ablation (BRS, V1, V1-t, V2, V2-t)
uv run python src/run_sweep.py --sweep ars_variants --budget 1600

# Hyperparameter sensitivity
uv run python src/run_sweep.py --sweep sigma_sensitivity --budget 3200
uv run python src/run_sweep.py --sweep alpha_sensitivity --budget 3200
uv run python src/run_sweep.py --sweep N_sensitivity --budget 3200

# Scaling study (LQR dim 4 → 256)
uv run python src/run_sweep.py --sweep lqr_scaling --budget 3200

# MuJoCo locomotion (requires gymnasium[mujoco])
uv run python src/run_sweep.py --sweep mujoco_comparison --budget 50000
```

### Two-phase evaluation protocol (paper §4)

```bash
# Phase 1: hyperparameter grid search (3 seeds per config)
uv run python src/sweep_protocol.py --phase grid \
    --tasks lqr pendulum --variants V2-t --results-dir results/protocol

# Phase 2: multi-seed evaluation with best config
uv run python src/sweep_protocol.py --phase eval100 \
    --best-configs results/protocol/best_configs.json \
    --n-seeds 100 --results-dir results/protocol
```

### Visualize results

```bash
uv run python src/visualize_all.py --results-dir results --out-dir results/figures
```

### Notebook

```bash
uv run jupyter notebook
# Open notebooks/ARS_Pendulum_Notebook.ipynb
```

---

## Repository Structure

```
stat4830/
├── src/                        # Core source code
│   ├── config.py               # ExperimentConfig, EnvConfig, MethodConfig dataclasses
│   ├── runner.py               # run_single(), run_sweep(), build_env(), build_method()
│   ├── run_sweep.py            # CLI: predefined sweep configs
│   ├── sweep_protocol.py       # Two-phase evaluation protocol (grid search → eval100)
│   ├── analysis.py             # episodes_to_threshold(), compute_seed_stats()
│   ├── visualize_all.py        # Plotting utilities for all sweep types
│   ├── policy.py               # LinearPolicy, RunningNorm (Welford online estimator)
│   ├── envs/
│   │   ├── base.py             # rollout(), eval_policy(), FrozenNorm
│   │   ├── lqr.py              # Linear Quadratic Regulator environment
│   │   ├── pendulum.py         # Pendulum (Gymnasium wrapper)
│   │   └── mujoco.py           # MuJoCo locomotion environments (optional)
│   └── methods/
│       ├── ars.py              # ARS: all variants (BRS, V1, V1-t, V2, V2-t)
│       ├── ars_ray.py          # Ray-parallel ARS (noise table in object store)
│       ├── vanilla_es.py       # Vanilla ES (one-sided perturbations)
│       ├── reinforce.py        # REINFORCE with mean baseline
│       └── base.py             # MethodResult dataclass
├── results/
│   ├── figures/                # Generated plots (PNG)
│   │   ├── full_comparison/
│   │   ├── ars_variants/
│   │   ├── lqr_scaling/
│   │   ├── sigma_sensitivity/
│   │   ├── alpha_sensitivity/
│   │   └── N_sensitivity/
│   └── notebook/               # Raw JSON result files (222 files)
│       ├── full_comparison/
│       ├── ars_variants/
│       ├── lqr_scaling/
│       ├── sigma_sensitivity/
│       ├── alpha_sensitivity/
│       ├── N_sensitivity/
│       └── protocol/           # Grid search + eval100 outputs
├── notebooks/
│   └── ARS_Pendulum_Notebook.ipynb
├── report.md                   # Week 6 report
├── self-critique.md            # Week 6 self-critique
├── pyproject.toml
└── CLAUDE.md
```

---

## ARS Variants

| Variant | Reward norm | State norm | Top-b selection |
|---------|:-----------:|:----------:|:---------------:|
| BRS     | ✗           | ✗          | b = N           |
| V1      | ✓           | ✗          | b = N           |
| V1-t    | ✓           | ✗          | b < N           |
| V2      | ✓           | ✓          | b = N           |
| V2-t    | ✓           | ✓          | b < N           |

---

## Ray Parallelism (optional)

For MuJoCo tasks where per-episode cost is high (~50–200ms), Ray parallelizes N rollout pairs across CPU cores:

```bash
uv run python src/run_sweep.py --sweep mujoco_comparison --budget 50000 \
    --use-ray --noise-table-size 1000000
```

Ray is not beneficial for LQR/Pendulum (episode cost ~1–5ms < Ray dispatch overhead).
