# Evolution Strategies for LLM Fine-Tuning: Mechanism Ablations and Population Scaling

**STAT 4830 — Week 10 Report**
**Date:** 2026-03-27

---

## 1. Problem Statement

### What are we optimizing?

We fine-tune frozen pre-trained language models (OPT-350M and OPT-1.3B) on **BoolQ** (yes/no question answering, 277 validation examples) using Evolution Strategies (ES) with no gradient access. The parameter vector $\theta \in \mathbb{R}^d$ is updated purely through scalar rewards from forward passes. Success is measured as peak validation accuracy under a fixed compute budget.

### Why does this problem matter?

Gradient-free fine-tuning is increasingly important for closed APIs, non-differentiable reward signals, and hardware-constrained settings. However, the ES design space — perturbation type, normalization, antithetic sampling, population size, selection strategy — is poorly characterized for LLMs. This week delivers the full mechanism ablation study across all five design dimensions, plus two targeted extensions: top-k selection at larger population sizes and population scaling on a 1.3B parameter model.

### How will we measure success?

- **Primary metric:** Mean peak validation accuracy across seeds under a fixed forward-pass budget.
- **Compute budget:** $B = N \times 2 \times \text{batch\_size} \times \text{iters} = 7{,}680$ training forward passes per variant. When $N$ varies, iterations adjust as $\lceil B / (N \times 2 \times 16) \rceil$.
- **X-axis:** Training forward passes (K) — validation overhead excluded. All curves start from the zero-shot baseline at $x = 0$.
- **Zero-shot baseline:** OPT-350M on BoolQ scores **0.094** before any ES training.

---

## 2. Technical Approach

### ES algorithm

Two-sided (antithetic) ES. For each iteration and population member $k = 1, \dots, N$:

$$
a_k = \frac{1}{B}\sum_{i \in \mathcal{B}} \text{reward}(\theta + \sigma \varepsilon_k, x_i) \;-\; \frac{1}{B}\sum_{i \in \mathcal{B}} \text{reward}(\theta - \sigma \varepsilon_k, x_i)
$$

$$
\hat{a}_k = \frac{a_k - \bar{a}}{\text{std}(a) + \epsilon}, \qquad \theta \leftarrow \theta + \frac{\alpha}{N} \sum_{k=1}^{N} \hat{a}_k \, \varepsilon_k
$$

**Calibrated hyperparameters** (Week 8): $\sigma = 3 \times 10^{-4}$, $\alpha = 10^{-3}$, batch = 16, train = 128, val = 277.

### Top-k selection

After evaluating all $N$ perturbations, only the $k$ members with the largest $|a_k|$ are used in the gradient update. $k = N$ recovers standard ES. **Design constraint:** z-score normalization is degenerate for $k \leq 2$ — with two samples, the normalized values are always exactly $\pm 1$ regardless of advantage magnitude, destroying scale information. Any experiment with $k \leq 2$ disables normalization.

---

## 3. Results

### 3.1 Mechanism ablations — OPT-350M on BoolQ (5 seeds)

#### One-sided vs. two-sided perturbation

![one_vs_two](../results/exp_boolq/one_vs_two/fig.png)

| Variant | Mean peak val_acc | Std |
|---|---|---|
| two\_sided | 0.529 | 0.004 |
| one\_sided | 0.529 | 0.003 |

Under a matched forward-pass budget (one-sided uses $2\times$ iterations), the two estimators converge identically. The antithetic pair reduces per-seed variance but provides no net benefit when averaged across seeds and iterations.

#### Noise type: Gaussian vs. Rademacher

![noise_type](../results/exp_boolq/noise_type/fig.png)

| Variant | Mean peak val_acc | Std |
|---|---|---|
| Gaussian | 0.529 | 0.004 |
| Rademacher | 0.527 | — |

No measurable difference. Both distributions have the same second moment ($\mathbb{E}[\varepsilon_i^2] = 1$) and the same ES gradient estimate in expectation, so the null result is theoretically predicted.

#### Normalization: z-score on vs. off

![normalize](../results/exp_boolq/normalize/fig.png)

| Variant | Mean peak val_acc | Std |
|---|---|---|
| normalized | 0.529 | 0.004 |
| unnormalized | 0.525 | 0.003 |

Normalization provides a small but consistent advantage — the unnormalized curve converges more slowly and to a lower asymptote. The learning curves show normalization accelerating early convergence before both plateau near the same region.

#### Population scaling — OPT-350M

![pop_scaling](../results/exp_boolq/pop_scaling/fig.png)

| N | Mean peak val_acc | Std |
|---|---|---|
| 4 | 0.527 | — |
| 8 | 0.529 | 0.004 |
| 16 | 0.530 | 0.003 |
| 32 | 0.533 | 0.012 |

Performance is nearly flat across N under fixed budget on OPT-350M. The slight upward trend at N=32 comes with higher variance — 5× more budget-efficient iteration steps versus a single large population update.

#### Top-k ablation — OPT-350M (N=16)

![top_k](../results/exp_boolq/top_k/fig.png)

| Variant | Mean peak val_acc | Std |
|---|---|---|
| all\_seeds ($k=16$) | 0.531 | 0.002 |
| top\_k = 4 | 0.530 | 0.004 |
| top\_k = 8 | 0.529 | 0.003 |

All three selection levels perform identically. Top-k filtering provides no benefit at N=16.

---

### 3.2 Top-k at N=8 — corrected analysis (3 seeds)

![top_k_corrected](../results_remote/top_k_corrected.png)

The N=8 ablation was initially run with top-k=2 using z-score normalization ON, producing a spuriously low result (mean=0.574) due to the normalization confound. The corrected experiment disables normalization for $k \leq 2$. All four variants now converge to the same accuracy band.

| Variant | Mean peak val_acc | Std | Seeds |
|---|---|---|---|
| all seeds ($k=8$) | **0.620** | 0.024 | [0.603, 0.603, 0.653] |
| top-k = 4 | 0.614 | 0.020 | [0.599, 0.599, 0.643] |
| top-k = 2 (no norm) | 0.599 | 0.033 | [0.570, 0.581, 0.646] |
| top-k = 1 (no norm) | 0.609 | 0.032 | [0.578, 0.596, 0.653] |

Even selecting a single perturbation per step ($k=1$) matches the full-population update. This is consistent with the fine-tuning landscape being low-dimensional and degenerate — filtering by advantage magnitude provides no additional signal concentration because all perturbations probe the same low-dimensional improvement subspace.

---

### 3.3 Top-k at N=64 (3 seeds)

![top_k_n64](../results_remote/exp_top_k_n64_20260326_192608/fig.png)

At N=64, the fixed budget allows only 4 gradient steps per variant. Even so, all k values converge to the same accuracy level, and top-k=1 (a single perturbation per update) performs equivalently to using all 64.

| Variant | Mean peak val_acc | Std |
|---|---|---|
| all seeds ($k=64$) | 0.603 | 0.036 |
| top-k = 32 | 0.615 | 0.025 |
| top-k = 16 | **0.620** | 0.025 |
| top-k = 8 | 0.614 | 0.028 |
| top-k = 1 (no norm) | 0.615 | 0.027 |

The null result at N=64 replicates the N=8 finding across a $8\times$ change in population size, strengthening the degeneracy interpretation.

---

### 3.4 Population scaling — OPT-1.3B (3 seeds)

![pop_scaling_1b_merged](../results_remote/pop_scaling_1b_merged.png)

![pop_scaling_1b_curves](../results_remote/pop_scaling_1b_curves.png)

Full population sweep on OPT-1.3B ($d \approx 1.3\text{B}$) under the same fixed 7,680 FP budget.

| N | Iters | Mean peak val_acc | Std | Notes |
|---|---|---|---|---|
| 1 | 240 | 0.622 | 0.023 | Rise-then-decay visible in curves |
| 2 | 120 | 0.297 | 0.152 | **Invalid** — normalization confound (rerun pending) |
| 4 | 60 | 0.460 | 0.055 | Underfitting — below baseline recovery |
| 8 | 30 | **0.621** | 0.023 | Best stable convergence |
| 16 | 15 | 0.611 | 0.028 | |
| 32 | 8 | 0.617 | 0.034 | |
| 64 | 4 | 0.537 | 0.101 | Only 4 gradient steps |
| 128 | 2 | 0.345 | 0.133 | Only 2 gradient steps |

**Key findings:**

- **N=1 rise-then-decay.** The single-perturbation curve peaks near 1–2K FPs then decays. This directly replicates the stiff/flat mode competition mechanism described by Liang et al. (2025): early training exploits high-curvature (stiff) directions rapidly; continued optimization then accumulates variance along flat directions, causing the decay. N=1 has the highest effective noise $\kappa = \sigma^2/N = \sigma^2$ and therefore fastest variance accumulation.

- **N=8 is the stable optimum.** With 8 perturbations, the per-step noise is reduced by $\sqrt{8}$ relative to N=1. The stiff-mode gain is captured without subsequent flat-mode variance accumulation. N=8 converges and holds, reaching the same peak as N=1 without the decay.

- **N=2 collapse is an artifact.** The N=2 run used z-score normalization, producing the same normalization confound as top-k=2. A corrected rerun with `--no-normalize` is scheduled.

- **N=4 underperforms.** With 60 iterations at N=4, the budget is sufficient but convergence is slower and peak accuracy is lower — suggesting N=4 is below the threshold where population averaging meaningfully reduces gradient noise for a 1.3B model.

- **N=64/128 are budget-exhausted.** With only 4 and 2 gradient steps respectively, these variants cannot approach meaningful accuracy. This is a budget allocation failure, not a fundamental property of large populations.

---

## 4. Next Steps

1. **Rerun N=2 on OPT-1.3B with `--no-normalize`** to fix the confounded data point and complete the scaling curve.
2. **Derive compute-optimal N\* allocation.** The N=1–32 curves allow fitting a power law $N^* \propto B^\alpha$ relating optimal population size to compute budget — a Chinchilla-style rule for ES.
3. **ES vs. Policy Gradient landscape comparison.** Run REINFORCE under matched compute budget to test whether both methods trace the same rise-then-decay trajectory on OPT-1.3B, directly addressing the original PG vs. ES framing.
4. **Estimate curvature-active dimension $d_\text{eff}$.** Log the per-iteration gradient covariance across the N=8 population and track its effective rank — the quantity that explains both the top-k null result and the rise-then-decay.
