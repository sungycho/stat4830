# Evolution Strategies for LLM Fine-Tuning: Calibration and Ablation Framework

**STAT 4830 — Week 8 Report**
**Date:** 2026-03-06

---

## 1. Problem Statement

### What are we optimizing?

We fine-tune a frozen pre-trained language model (OPT-350M / OPT-1.3B) for binary classification tasks using Evolution Strategies (ES). The model is treated as a black box: we have no access to gradients, only scalar rewards from completed forward passes.

Concretely, the parameter $\theta \in \mathbb{R}^d$ ($d \approx 330M$ for OPT-350M) is updated to maximize accuracy on:

- **RTE (Recognizing Textual Entailment):** Binary entailment prediction from a premise–hypothesis pair. 277 validation examples (full GLUE validation split).
- **BoolQ:** Yes/no question answering conditioned on a reading passage. 500 validation examples.

Both tasks score $+1$ for a correct first-token prediction matching the gold label, $-1$ otherwise (reward $\in \{-1, +1\}$). Mini-batch advantages $r = \frac{1}{B}\sum_{i=1}^B \text{reward}_i \in [-1, +1]$.

### Why does this problem matter?

Fine-tuning large language models requires gradient access, which is often unavailable (closed APIs, hardware constraints, non-differentiable reward signals). Zero-order methods that query the model purely through forward passes are a natural alternative. However, the mechanism design space is large: perturbation type, update rule normalization, antithetic sampling, and population size all affect both performance and compute efficiency. There is no systematic understanding of which components matter most in the LLM fine-tuning regime (high dimension, binary reward, short generation).

This study builds toward a controlled empirical dissection of ES mechanisms framed as: *which algorithmic choices give the best val_acc per training forward pass?* This week's scope is the prerequisite step — calibrating $\sigma$ and $\alpha$ on RTE — which must be completed before any mechanism comparison is meaningful.

### How will we measure success?

- **Primary metric (calibration):** Peak validation accuracy on the RTE held-out split across the sigma/lr grid.
- **X-axis for plots:** Cumulative training forward passes — forward passes spent on perturbation evaluation only, not counting validation overhead. This is the controlled compute budget.
- **Baseline:** Zero-shot (untuned) accuracy of OPT-350M on RTE, measured before any ES updates.

### Scope for this deliverable

Due to compute time constraints, this week focuses on the **calibration phase only**: a 4×4 grid search over $(\sigma, \alpha)$ on RTE with OPT-350M. The full mechanism ablation (one-sided vs. two-sided, Gaussian vs. Rademacher, normalization, top-k, population scaling) is designed, implemented, and ready to run, but deferred to the next deliverable once calibrated hyperparameters are in hand.

---

## 2. Technical Approach

### Mathematical formulation

**Two-sided (antithetic) ES** with z-score normalized rewards. For each iteration:

$$
\text{Sample batch } \mathcal{B} \subset \mathcal{D}_{\text{train}}, \quad |\mathcal{B}| = B.
$$

For $k = 1, \dots, N$: draw seed $s_k$; reconstruct noise $\varepsilon_k \sim p_\varepsilon$ from $s_k$:

$$
r_k^+ = \frac{1}{B}\sum_{i \in \mathcal{B}} \text{reward}(\theta + \sigma \varepsilon_k, x_i), \qquad
r_k^- = \frac{1}{B}\sum_{i \in \mathcal{B}} \text{reward}(\theta - \sigma \varepsilon_k, x_i).
$$

$$
a_k = r_k^+ - r_k^- \in [-2, 2].
$$

Normalize advantages (z-score):

$$
\hat{a}_k = \frac{a_k - \bar{a}}{\text{std}(a) + \epsilon}.
$$

Update:

$$
\theta \leftarrow \theta + \frac{\alpha}{N} \sum_{k=1}^N \hat{a}_k \, \varepsilon_k.
$$

**One-sided ES** replaces the antithetic pair with a single evaluation: $a_k = r_k^+$. Uses $2\times$ more iterations to match the training forward-pass budget.

**ARS-style top-$k$** selects the subset of seeds $\mathcal{S}_k \subset \{1,\dots,N\}$ with the $k$ largest $|a_j|$ before normalizing and updating.

**Rademacher noise:** $\varepsilon_{ki} \in \{-1, +1\}$ independently and uniformly, instead of $\mathcal{N}(0, I)$. Variance is identical ($\mathbb{E}[\varepsilon_i^2] = 1$) but the distribution is bounded, which may reduce variance in the reward difference estimator.

### Key implementation design: seed-based in-place perturbation

Storing $d \approx 330M$ noise vectors per seed would require $\sim 1.2$ GB each (float32). Instead, we follow the LAES/EvoLLM design:

1. Only the integer seed is stored, never the noise vector.
2. Each perturbation reconstructs noise on-the-fly from the seed, advancing a single `torch.Generator` across all layers in a fixed order.
3. Peak extra memory = size of the largest single parameter tensor (typically the embedding or output projection).
4. Roundtrip invariance: restoring parameters to $\theta$ from $\theta \pm \sigma\varepsilon$ is exact (same generator, reversed sign).

The three-pass antithetic trick avoids a fourth generator pass per seed:

$$
\theta \xrightarrow{+\sigma\varepsilon_k} (\theta + \sigma\varepsilon_k) \xrightarrow{-2\sigma\varepsilon_k} (\theta - \sigma\varepsilon_k) \xrightarrow{+\sigma\varepsilon_k} \theta
$$

This requires 3 parameter scans per seed (vs. 4 in the naive approach) but no model copies and no stored noise.

### Implementation files

| File | Role |
|---|---|
| `src/utils/perturb.py` | `perturb_inplace`, `restore_inplace`, `es_grad_update` with Rademacher/top-k/normalize flags |
| `src/backends/hf_backend.py` | Batched generation via `generate_batch`; left-padding for causal LMs |
| `src/tasks/{sst2,rte,boolq}.py` | Task registry; `build_prompt`, `score`, `load_data` |
| `src/scripts/sanity_es_loop.py` | Minimal CPU-runnable ES loop (single-process, readable, no GPU deps) |
| `src/scripts/train_es.py` | GPU-ready training script; all flags CLI-configurable; forward-pass tracking |
| `src/scripts/run_experiment.py` | Orchestrates all experiment blocks; fixed-budget pop scaling; per-variant seed runs |
| `src/scripts/plot_results.py` | Plots val_acc vs training forward passes; seed interpolation to common grid; calibration heatmap |

### Forward-pass budget accounting

Two quantities are tracked separately:

- `train_fwd`: forward passes on perturbation evaluation only (the controlled budget, used as x-axis).
- `total_fwd`: `train_fwd` + validation passes (logged for reference, not plotted).

For two-sided ES: $\text{train\_fwd per iter} = N \times 2 \times B$.
For one-sided ES: $\text{train\_fwd per iter} = N \times 1 \times B$.

Validation is run every `val_every` iterations and costs `|val\_set|` passes, which are charged to `total_fwd` only.

### Experiment blocks

The full ablation framework is implemented and ready; only the calibration block runs this week.

| Block | Variants | Budget matching | Status |
|---|---|---|---|
| **Calibration** | $\sigma \in \{3\times10^{-4}, 10^{-3}, 3\times10^{-3}, 10^{-2}\} \times \alpha \in \{10^{-4}, 3\times10^{-4}, 10^{-3}, 3\times10^{-3}\}$ | Fixed iters=20, pop=8 | **This week** |
| One vs. two-sided | two\_sided (30 iters), one\_sided (60 iters) | Equal train\_fwd = 7680 | Deferred |
| Noise type | Gaussian, Rademacher | Equal | Deferred |
| Normalize | z-score on, z-score off | Equal | Deferred |
| Top-k | all (N=16), top-4, top-8 | Equal; top-k applied at update only | Deferred |
| Population scaling | $N \in \{4, 8, 16, 32\}$, iters = $\lceil 7680/(N\cdot2\cdot B)\rceil$ | Equal train\_fwd $\approx 7680$ | Deferred |

---

## 3. Results

### 3.1 Zero-shot baseline

| Model | Task | Zero-shot val_acc |
|---|---|---|
| OPT-350M | RTE | 0.455 |

### 3.2 Calibration: sigma × lr grid on RTE

Full 4×4 grid over $\sigma \in \{3\times10^{-4}, 10^{-3}, 3\times10^{-3}, 10^{-2}\}$ and $\alpha \in \{10^{-4}, 3\times10^{-4}, 10^{-3}, 3\times10^{-3}\}$, 3 seeds each, 20 iterations, population=8, batch=16, train=128.

| Rank | $\sigma$ | $\alpha$ | Mean peak val_acc | Std | $\Delta$ zero-shot |
|---|---|---|---|---|---|
| 1 | $3\times10^{-4}$ | $10^{-3}$ | **0.5331** | 0.0045 | +7.8% |
| 2 | $10^{-3}$ | $3\times10^{-3}$ | 0.5331 | 0.0061 | +7.8% |
| 3 | $3\times10^{-4}$ | $3\times10^{-3}$ | 0.5283 | 0.0017 | +7.3% |
| 3 | $3\times10^{-3}$ | $3\times10^{-4}$ | 0.5283 | 0.0017 | +7.3% |
| … | … | … | … | … | … |
| 13–16 | $10^{-2}$ | any | 0.4958 | 0.0274 | −4.2% |

**Best config: $\sigma = 3\times10^{-4}$, $\alpha = 10^{-3}$** — carried forward to all ablation blocks.

**Key observations:**

- **ES produces meaningful gains.** The best config reaches 0.533, a +7.8% improvement over the zero-shot baseline of 0.455. Even mid-table configs around 0.527 represent a +7.2% gain, indicating that ES is reliably learning across a range of hyperparameters.
- **$\sigma = 10^{-2}$ is the only regime that degrades below zero-shot.** All four lr values at the largest sigma yield mean val_acc of 0.496 (−4.2% vs. zero-shot). Large perturbations destroy model coherence before the update can correct it; the advantage signal becomes dominated by generation quality collapse rather than task accuracy.
- **Low lr ($\alpha = 10^{-4}$) is noisy and unreliable.** Configs with $\alpha = 10^{-4}$ show the highest variance (std up to 0.027) and lowest means, suggesting steps are too small to accumulate signal within 20 iterations.
- **The effective operating range is moderately broad but has a hard upper boundary.** Reliable improvement is achieved across $\sigma \in \{3\times10^{-4}, 10^{-3}, 3\times10^{-3}\}$ and $\alpha \in \{10^{-3}, 3\times10^{-3}\}$, but $\sigma = 10^{-2}$ is a hard failure mode.

---

## 4. Next Steps

### Immediate (next deliverable)

1. **Run all five ablation blocks** with calibrated HPs and 3 seeds each:
   - One-sided vs. two-sided ES
   - Gaussian vs. Rademacher noise
   - Normalized vs. unnormalized update
   - ES vs. ARS-style top-k (N=16, top-4/8)
   - Population scaling $N \in \{4, 8, 16, 32\}$ at fixed budget
2. **Confirm best config on BoolQ** via `--block task_confirm --best-sigma ... --best-lr ...`
4. **Populate §3** with numbers and val_acc vs. train_fwd plots.

### Open questions these ablations will answer

- Does Rademacher noise help for binary {−1, +1} rewards where the reward Hessian has few large eigenvalues?
- Does top-k selection help when most seeds give near-zero advantage (all-correct or all-wrong batches)?
- Does the optimal population size track the effective rank of the relevant parameter subspace?
- Is z-score normalization critical for stability, or does the bounded advantage range ($a_k \in [-2, 2]$) make it redundant?

### GPU execution

See `notebooks/week8_implementation.ipynb` for step-by-step setup and all run commands.
