# STAT 4830 Project: Evolution Strategies for Black-Box LLM Fine-Tuning

This repository studies **Evolution Strategies (ES)** as a gradient-free method for fine-tuning large language models. The core question: *which ES algorithmic choices matter most when optimizing LLMs with only scalar reward feedback?*

The project evolved in two phases:
1. **Weeks 4–6:** Reproduced ARS (Mania et al., NeurIPS 2018) on classic control tasks (LQR, Pendulum), benchmarked against Vanilla ES and REINFORCE
2. **Weeks 8–9:** Pivoted to black-box LLM fine-tuning — applied ES to OPT-350M on NLP classification tasks (RTE, BoolQ) with a systematic ablation framework

---

## Key Results

### Phase 1 — Classic Control (LQR + Pendulum)

| Method | LQR final return | Pendulum final return |
|--------|----------------:|-----------------------:|
| ARS V2-t | **−16.6** | **−1,162** |
| Vanilla ES | −179,104 | −1,734 |
| REINFORCE | −178,106 | −1,601 |

- ARS converges ~10,800× better than baselines on LQR
- Reward normalization alone (BRS → V1) accounts for a 14,600× improvement

### Phase 2 — LLM Fine-Tuning (OPT-350M on BoolQ)

Calibrated HPs: σ=3e-4, α=1e-3 (from 4×4 grid search on RTE, 5 seeds)

| Ablation | Finding |
|---|---|
| **Normalization** | Most impactful — z-score normalization converges to 0.62 vs 0.54 unnormalized, with far lower variance |
| **Top-k selection** | top-k=4 (out of N=16) is unstable and degrades over time; top-k=8 is safe |
| **One-sided vs two-sided** | No meaningful difference at matched forward-pass budget |
| **Noise type** | Gaussian vs Rademacher converge to same accuracy; Rademacher slightly faster early |
| **Population scaling** | Larger N is more stable; diminishing returns within fixed budget |

Zero-shot baseline: 0.451 (RTE), ~0.62 achieved after ES fine-tuning (BoolQ)

---

## Setup

```bash
# Python 3.13, managed with uv
uv sync
source .venv/bin/activate
```

GPU required for LLM experiments (≥2GB VRAM for OPT-350M in bfloat16).

---

## Running Experiments

### Phase 1 — Classic Control

```bash
# Full method comparison (ARS vs Vanilla ES vs REINFORCE)
uv run python src/run_sweep.py --sweep full_comparison --budget 3200

# ARS variant ablation
uv run python src/run_sweep.py --sweep ars_variants --budget 1600

# Visualize
uv run python src/visualize_all.py --results-dir results --out-dir results/figures
```

### Phase 2 — LLM Fine-Tuning

```bash
# Calibration grid search (run first)
uv run python -m src.scripts.run_experiment --block calibration --device cuda --dtype bfloat16 --n-seeds 5 --out-dir results/exp_calibration

# Ablation blocks (use calibrated HPs)
uv run python -m src.scripts.run_experiment --block one_vs_two --device cuda --dtype bfloat16 --n-seeds 3 --best-sigma 3e-4 --best-lr 1e-3 --task boolq --out-dir results/exp_boolq

# Available blocks: one_vs_two, noise_type, normalize, top_k, pop_scaling, task_confirm

# Visualize all blocks
uv run python -m src.scripts.plot_results --exp-dir results/exp_boolq --recursive
```

---

## Repository Structure

```
stat4830/
├── src/
│   ├── scripts/                    # LLM ES fine-tuning pipeline
│   │   ├── train_es.py             # GPU-ready ES training loop
│   │   ├── run_experiment.py       # Ablation block orchestrator
│   │   ├── plot_results.py         # val_acc curves + calibration heatmap
│   │   └── sanity_es_loop.py       # CPU-runnable reference implementation
│   ├── utils/
│   │   └── perturb.py              # Seed-based in-place perturbation (no stored noise)
│   ├── backends/
│   │   └── hf_backend.py           # HuggingFace batched inference
│   ├── tasks/                      # RTE, BoolQ task registry
│   ├── envs/                       # LQR, Pendulum environments (Phase 1)
│   └── methods/                    # ARS, Vanilla ES, REINFORCE (Phase 1)
├── results/
│   ├── exp_boolq/                  # BoolQ ablation results
│   ├── exp_week9/                  # RTE ablation results
│   └── week6-figures/              # Classic control plots
├── notebooks/
│   ├── week8_implementation.ipynb  # LLM ES walkthrough
│   └── ARS_Pendulum_Notebook.ipynb # Phase 1 notebook
└── .report/                        # Weekly reports and self-critiques
```

---

## Design Highlights

- **Seed-based in-place perturbation:** Noise reconstructed on-the-fly from integer seeds — no stored noise vectors, no model copies. Peak memory overhead = size of largest single layer (~50MB for OPT-350M)
- **Budget-controlled comparison:** All ablation variants matched on training forward passes (not wall-clock time or iterations)
- **Three-pass antithetic trick:** θ → θ+σε → θ−σε → θ in 3 parameter scans per seed vs 4 in the naive approach
