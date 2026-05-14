# Reward-Aware Population Scaling of Evolutionary Strategies in LLM Fine-Tuning

**STAT 4830 — Spring 2026, University of Pennsylvania**
Sung Cho · Gyubin Han · Maxwell DeLorenzo

Source-of-truth artifacts: [`.final-deliverable/final-report.pdf`](.final-deliverable/final-report.pdf) (theoretical, ICM-workshop submission) and [`.final-deliverable/final-slides.pdf`](.final-deliverable/final-slides.pdf) (broader experimental story).

---

## Tl;dr

We study Evolutionary Strategies (ES) as a zeroth-order optimizer for fine-tuning frozen LLMs through scalar rewards. The literature reports two seemingly contradictory recommendations for population size `N`:

| Paper | Reward | Recommended N |
|---|---|---|
| MeZO (Malladi et al., 2023) | Cross-entropy (white-box) | `N = 1` |
| ES-at-Scale (Qiu et al., 2026) | Binary accuracy (black-box) | `N ≈ 30` |

We argue this is **not a paradox** but a scaling law: the right `N` is set by the **reward function**, not by the model or task. Concretely:

1. **Reward granularity sets an availability threshold `N_avail`.** Under CE reward, every perturbation pair carries signal almost surely (`N_avail = 1`). Under binary accuracy reward, a positive fraction `q = P(A_acc = 0)` of pairs is **degenerate** (both perturbations give the identical batch score), and any usable update needs `N ≥ N_avail(δ) = ⌈log δ / log q⌉`.

2. **Advantage normalization — not population size — is what kills small-`N` binary ES.** Z-score normalization erases the reward-scale information that small populations *need* to behave like the raw (self-annealing) estimator. With normalization off, an `N = 2` binary-reward run on Qwen2.5-1.5B / GSM8K and TREC recovers and *outperforms* the standard normalized variant.

A one-shot **zero-training degeneracy probe** estimates `N_avail` from `2KB` forward passes (no training); we validate it on Qwen2.5-Instruct (0.5B/1.5B/7B) × GSM8K across twelve configurations to mean absolute error 0.020.

The slides additionally develop a **benign-vs-malignant taxonomy** (preliminary): the vanilla model's *failure mode* — not raw accuracy — predicts whether ES can learn at all. A single statistic, the intra-pair correctness correlation `ρ`, lands in a moderate band (≈ 0.3–0.7) precisely when ES is viable.

---

## Table of contents

1. [Background: ES for LLM fine-tuning](#1-background-es-for-llm-fine-tuning)
2. [The population mystery (Lesson 1)](#2-the-population-mystery-lesson-1)
3. [Vanilla failure mode predicts performance (Lesson 2 — slides)](#3-vanilla-failure-mode-predicts-performance-lesson-2--slides)
4. [Normalization sometimes hurts (Lesson 3)](#4-normalization-sometimes-hurts-lesson-3)
5. [Theoretical contributions in one place](#5-theoretical-contributions-in-one-place)
6. [Repository layout](#6-repository-layout)
7. [Setup](#7-setup)
8. [Reproducing the results](#8-reproducing-the-results)
9. [Project history — how we got here](#9-project-history--how-we-got-here)
10. [Limitations](#10-limitations)
11. [References](#11-references)

---

## 1. Background: ES for LLM fine-tuning

For parameters `θ ∈ ℝ^d`, ES estimates the gradient of the Gaussian-smoothed objective `f_σ(θ) = 𝔼_ε[f(θ + σε)]` through antithetic perturbations:

```
ĝ = (1 / Nσ) Σ_{i=1..N} [R(θ + σε_i, B) − R(θ − σε_i, B)] ε_i,    ε_i ~ N(0, I_d)
θ ← θ + η ĝ
```

Three properties make ES attractive for LLM fine-tuning:

- **Memory efficiency.** No activations are stored: cost is `O(d)`, not `O(d · seq_len)`.
- **Distributable.** Each seed is independent — embarrassingly parallel.
- **Black-box compatible.** The reward `R(·)` need not be differentiable. ES is the only viable fine-tuning method when only scalar outputs (correct/incorrect, preference ratings) are available.

The headline objection is the curse of dimensionality: variance scales as `d/N`. Liang et al. (2026) resolved this with the *blessing of dimensionality*: the effective rank `r` of the loss Hessian is `O(100)` for pretrained LLMs, so `N ≤ r` perturbations already span the useful subspace. That answers the *upper* end of `N`. Our work answers the *lower* end.

---

## 2. The population mystery (Lesson 1)

> **Different rewards demand different populations.**

Two papers fine-tuned LLMs with literally the same ES estimator and reached opposite conclusions about `N`:

```
MeZO  (CE  reward, white-box)   →  N = 1 suffices
ES-at-Scale (binary, black-box) →  N ≈ 30 required
```

We show this is a function of the reward signal.

**Availability threshold.** Let `K_N = #{i : A_i ≠ 0}` count the non-degenerate seeds in a population of size `N`, where `A_i = R(θ + σε_i, B) − R(θ − σε_i, B)`. Setting `q = P(A_i = 0)`:

```
P(K_N = 0) = q^N,        N_avail(δ) = ⌈log δ / log q⌉
```

`N_avail` is the smallest population for which at least one informative seed appears with probability `≥ 1 − δ`. Below `N_avail` the ES update typically carries no reward-bearing direction at all and training stalls or degrades.

**CE limit.** For continuous CE reward, `q = 0` almost surely (the zero set of a non-identically-zero real-analytic function in `ε` is Lebesgue-null), so `N_avail = 1`. Under appropriate learning-rate scaling `η_N = N η_0`, the cumulative drift and diffusion over `T = M/N` steps are *independent of `N`* (Proposition 1, "CE population indifference"). MeZO's `N = 1` is not lucky — it's a consequence of CE reward.

**Binary regime.** Binary accuracy reward induces a closed-form degeneracy probability:

```
q ≈ 1 / √(4π B p_0 (1 − p_0)(1 − ρ))
```

where `p_0` is the base model's accuracy and `ρ` is the intra-pair correctness correlation. The full derivation (Proposition 3 + Appendix D) is a local-CLT / Edgeworth expansion with vanishing `O(B^{−1/2})` Berry–Esseen term thanks to `Y_j ∈ {−1, 0, +1}` having zero third moment.

**Empirical probe.** A zero-training forward-pass probe (Appendix E) estimates `q̂, p̂_0, ρ̂` from `2KB` forward passes. On Qwen2.5-Instruct (0.5B/1.5B/7B) × GSM8K, across 12 configurations the formula matches the empirical `q` with mean absolute error 0.020. For Qwen2.5-1.5B at `B = 16`, `q̂ ≈ 0.19`, so `N_avail ≤ 4` — small enough that any `N ≥ 4` is above threshold. The empirically-reported `N ≈ 30` floor in ES-at-Scale reflects their lower-`p_0` regime, not a property of LLMs in general.

---

## 3. Vanilla failure mode predicts performance (Lesson 2 — slides)

> **The type of error a pre-trained model makes — not its raw accuracy — determines whether ES can succeed at all.**

This lesson is the slides' main empirical conjecture and goes *beyond* the formal report. It splits binary-reward ES into two regimes:

| | **ES-Benign** | **ES-Malignant** |
|---|---|---|
| Error type | Incorrect but task-aware | Fundamental incapacity |
| What it looks like | Reasoning steps present, wrong final number | Says "no" to everything; repeats the prompt |
| Why | Format understood, knowledge/arithmetic missing | Model can't engage with the task at all |
| Correctable by ES? | Yes — errors share a direction across examples | No — errors are structurally irreducible |

We use the intra-pair correctness correlation `ρ = Corr(R(θ + σε), R(θ − σε))` as a numerical diagnostic:

| | LLaMA-1B / MNLI | Qwen-1.5B / GSM8K | Qwen-1.5B / WIC |
|---|---|---|---|
| `ρ` | 0.14 | 0.42 | 0.88 |
| `ρ` regime | Very low | Moderate | Very high |
| Failure mode | Chaotic — repeats prompt | Capable-but-imperfect | Frozen — says "no" |
| Empirical ES outcome | Malignant | **Benign** | Malignant |

ES is viable in the moderate band `ρ ≈ 0.3–0.7`; both extremes (random chaos, frozen consensus) collapse to no useful gradient. The malignant regime is consistent with our earlier Week 10 finding that even `top-k = 1` matches the full population on BoolQ — a sign the surviving useful subspace is effectively low-dimensional regardless of `N`.

---

## 4. Normalization sometimes hurts (Lesson 3)

> **z-score advantage normalization can — by itself — convert a viable `N = 2` binary-reward run into a collapse.**

The default ES recipe normalizes pair-level advantages by their empirical standard deviation:

```
Â_i = (A_i − Ā) / (s_A + ε_0)
```

At `N = 2`, this *erases the absolute advantage gap*. For two pair-level advantages `A_1, A_2`, the centered, scale-normalized vector `(±1, ∓1)` is independent of `|A_1 − A_2|` (Proposition 4). An infinitesimal advantage gap and a large one produce the same normalized update magnitude. With raw advantages, ES is *self-annealing* — updates shrink as advantages shrink near degeneracy. Normalization removes that brake.

**Single fixed-everything ablation (Qwen2.5-1.5B, 3 seeds 42/43/44):**

| Setting | GSM8K final val_acc | TREC final val_acc |
|---|---|---|
| Normalize on (default) | ≈ 0.02 (collapses) | ≈ 0.30 (degrades) |
| **Normalize off** | **≈ 0.55** | **≈ 0.70** |

(Figure 1 of `final-report.pdf`.) The apparent need for large populations under binary reward is, in this regime, an implementation artifact — not a property of the LLM or task. A clipped-standard-deviation rule `Â_i = (A_i − Ā) / max(s_A, ε_0)` is one proposed compromise that retains scale control while preventing the small-`N` amplification.

---

## 5. Theoretical contributions in one place

| Claim | Statement | Where |
|---|---|---|
| **Prop 1 (CE population indifference)** | Under CE reward and `η_N = N η_0`, cumulative drift and diffusion across `T = M/N` steps are `N`-independent. | report §3 |
| **Cor 2 (No availability threshold under CE)** | `K_N = N` almost surely whenever `∇_θ R_CE ≠ 0`. | report §3 |
| **Prop 3 (Binary degeneracy approximation)** | `q ≈ 1 / √(4πB p_0 (1−p_0)(1−ρ))`. | report §3, Appx D |
| **Prop 4 (z-score erases advantage scale at `N = 2`)** | `((A_1 − Ā)/s_A, (A_2 − Ā)/s_A)` is independent of `\|A_1 − A_2\|`. | report §4, Appx G |
| **Thm (Nesterov–Spokoiny `L_σ`-smoothness)** | `f_σ` is `L_σ`-smooth with `L_σ = 2/σ²` even when `f` is discontinuous (e.g. binary reward). | report Appx B |
| **N-cancellation (Week 12 report)** | At fixed budget `B_total = N · T` and `η ∝ N`, total progress over `T = B/N` steps is independent of `N` for `N ∈ [N_min, r]`. | Week 12 §5 |
| **Sandwich bound (Week 12 report)** | `N_min ≤ N* ≤ r`; practical recommendation is `N* = N_min`. | Week 12 §6 |
| **Spectral-decay resolution of the stability gap** | The empirical `η ∼ 10⁻³–10⁻⁴` regime is consistent with `L_σ^true ≈ 2r / (d(β−1)σ²)` for power-law spectra `λ_k ∝ k^{−β}`. | report Appx H |

A dense narrative reading: read sections 1–5 of `final-report.pdf` for the core ICM-workshop argument; appendices A–H for the formal proofs; sections 1–10 of `.report/Week_12_Report.pdf` for the earlier full SNR + N-cancellation development (later condensed into the workshop version).

---

## 6. Repository layout

```
stat4830/
├── .final-deliverable/     ← source of truth (final-report.pdf, final-slides.pdf, demo.ipynb)
├── .report/                ← bi-weekly progress reports (week4 / 6 / 8 / 10 / 12)
├── src/
│   ├── scripts/
│   │   ├── train_es.py             ← main ES fine-tuning entry point
│   │   ├── run_experiment.py       ← orchestrates ablation blocks
│   │   ├── analyze_failures.py     ← failure-mode classification utilities
│   │   ├── plot_results.py
│   │   └── adhoc/                  ← experiments, probes, and plotting helpers
│   │       ├── probe_degeneracy.py        ← Appendix E pre-training probe
│   │       ├── run_rho_sweep.py           ← multi-model × multi-task ρ sweep
│   │       ├── run_norm_n2.py             ← N=2 norm on vs off driver
│   │       ├── run_pop_sweep_mnli.py
│   │       ├── plot_norm_n2_sweep.py
│   │       ├── plot_pop_scaling_3seeds.py
│   │       ├── plot_gsm8k_norm_comparison.py
│   │       └── …
│   ├── tasks/              ← {sst2, rte, boolq, mnli, cb, wsc, wic, copa, trec,
│   │                          sst5, squad, drop, record, gsm8k, math500, countdown}
│   ├── backends/           ← HF causal-LM batched generation
│   ├── parsing/, prompting/, rollout/
│   └── utils/perturb.py    ← seed-based in-place ±σε perturbation (3-pass trick)
├── scripts/                ← shell drivers (run_norm_n2_batch.sh, …)
├── notebooks/              ← legacy notebooks (ARS pendulum, PG vs ES)
├── configs/, docs/, legacy/, results/, results_remote/, plots/, tests/
├── pyproject.toml          ← uv-managed Python 3.13 project
└── README.md               ← this file
```

The core ES update (`src/utils/perturb.py`) never materializes a noise vector: it advances a single `torch.Generator` from a stored integer seed and walks the parameters layer-by-layer, so peak extra memory equals the size of the largest parameter tensor.

---

## 7. Setup

```bash
# 1. Python 3.13 + uv
uv sync

# 2. (optional) activate the venv for interactive use
source .venv/bin/activate

# 3. (HF-gated models) authenticate once
huggingface-cli login
```

Most experiments assume a single CUDA GPU. The probe and small-model ablations run on a 24 GB card; ablations on Qwen2.5-7B or OPT-13B need 80 GB-class hardware (we used `ubuntu@216.81.245.44`-class single-GPU nodes).

---

## 8. Reproducing the results

The recipes below are grouped by which figure / table of `final-report.pdf` and `final-slides.pdf` they generate. All commands are run from the repo root.

### 8.1 Zero-training degeneracy probe (Table 1 / Appendix E)

Empirically estimates `q̂ = P(A = 0)`, `p̂_0`, `ρ̂`, and `N_avail(δ = 0.05)` from `2KB` forward passes. No training.

```bash
# Qwen2.5-1.5B-Instruct on GSM8K, σ = 1e-3, K = 200 pairs, B = 16
uv run python -m src.scripts.adhoc.probe_degeneracy \
    --task gsm8k \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --sigma 1e-3 --K 200 --batch-size 16

# Faster: 4 model copies on an 80 GB GPU
uv run python -m src.scripts.adhoc.probe_degeneracy \
    --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct \
    --K 200 --batch-size 16 --num-workers 4
```

### 8.2 Multi-model × multi-task ρ sweep (slides Lesson 2)

Runs the probe over the full registry and produces the `ρ` / failure-mode table used for the benign/malignant taxonomy:

```bash
# Full sweep (every model × every task) — long
uv run python -m src.scripts.adhoc.run_rho_sweep --K 200

# Targeted: only the slide-table models
uv run python -m src.scripts.adhoc.run_rho_sweep \
    --models llama3.2-1b qwen2.5-1.5b-instruct \
    --tasks  mnli gsm8k wic --K 200

# Resume an interrupted sweep
uv run python -m src.scripts.adhoc.run_rho_sweep --skip-done
```

Aggregate the resulting JSON outputs with `src/scripts/adhoc/classify_rho_buckets.py` and `table_rho_p0.py`.

### 8.3 ES fine-tuning with `train_es.py` (slides Lesson 1, both PDFs)

This is the workhorse script. Defaults are the calibrated `σ = 1e-3`, `η = 1e-4`, batch 16, val 200; everything is CLI-configurable.

```bash
# Smoke test on SST-2 (matches the example slide for OPT-13B / SST-2)
uv run python -m src.scripts.train_es \
    --task sst2 --model facebook/opt-1.3b \
    --population-size 8 --num-iters 50 --sigma 1e-3 --lr 1e-4

# Qwen2.5-1.5B / GSM8K population-scaling, one seed of one population
uv run python -m src.scripts.train_es \
    --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
    --population-size 8 --num-iters 240 \
    --batch-size 16 --train-size 128 --val-size 200 \
    --sigma 1e-3 --lr 1e-3 --reward accuracy \
    --seed 42 --save --out-dir results/gsm8k_n8_s42

# CE reward (white-box) variant — used to reproduce CE population indifference
uv run python -m src.scripts.train_es \
    --task sst2 --model facebook/opt-1.3b \
    --reward ce --population-size 1 --num-iters 600 \
    --sigma 1e-3 --lr 1e-4

# One-sided estimator
uv run python -m src.scripts.train_es --task boolq --one-sided ...

# Rademacher noise instead of Gaussian
uv run python -m src.scripts.train_es --task boolq --noise-type rademacher ...

# ARS-style top-k filtering
uv run python -m src.scripts.train_es --task boolq --top-k 4 ...
```

For the population-scaling curves (final-report Figure 2b; slides "Binary reward does NOT work for all N"), sweep `--population-size` over `{1, 2, 4, 8, 16, 32}` while holding the total forward-pass budget fixed: `num_iters = ⌈B_total / (N · 2 · batch_size)⌉`. Average across `--seed 42 43 44` and plot with `plot_pop_scaling_3seeds.py`.

### 8.4 N=2 normalization recovery (Figure 1 of `final-report.pdf`)

The single ablation that turns the slides' "small-N collapse" into a "small-N can be best" — the central empirical result of the report.

```bash
# Direct, single (model, task) pair, 3 seeds
uv run python -m src.scripts.adhoc.run_norm_n2 \
    --model qwen2.5-1.5b-instruct --task gsm8k \
    --seeds 42 43 44 \
    --lr-on 1e-3 --lr-off 1e-3 --sigma 1e-3

# Same but for the TREC panel of Figure 1
uv run python -m src.scripts.adhoc.run_norm_n2 \
    --model qwen2.5-1.5b-instruct --task trec \
    --seeds 42 43 44

# Multi-(model, task) batch wrapper (see scripts/run_norm_n2_batch.sh
# for the full PAIRS array)
bash scripts/run_norm_n2_batch.sh

# Heavier generation task — careful, long
bash scripts/run_norm_n2_math500.sh
```

Plot the resulting sweep directory:

```bash
uv run python -m src.scripts.adhoc.plot_norm_n2_sweep \
    --exp-dir results/norm_n2_qwen2.5-1.5b-instruct_gsm8k_<timestamp> \
    --title "Qwen2.5-1.5B — GSM8K N=2: normalize on vs off"

# Or plot every results/norm_n2_*/ at once
uv run python -m src.scripts.adhoc.plot_norm_sweep --all
```

### 8.5 GSM8K population scaling and norm-comparison figures

```bash
# Slides figure: GSM8K population scaling for N ∈ {1, 2, 4, 8, 16}, 3 seeds
uv run python -m src.scripts.adhoc.plot_pop_scaling_3seeds

# Final-report Figure 1(a): N=2 normalize on vs off on GSM8K
uv run python -m src.scripts.adhoc.plot_gsm8k_norm_comparison
```

### 8.6 Other useful entry points

```bash
# Hyperparameter / LR sweep on MNLI (used for the slides' Llama-3.2-1B figure)
uv run python -m src.scripts.adhoc.run_lr_sweep_mnli --help
uv run python -m src.scripts.adhoc.plot_lr_sweep_mnli ...

# Population sweep on MNLI (slides Llama N_min panel)
uv run python -m src.scripts.adhoc.run_pop_sweep_mnli --help

# Failure-mode decomposition (correct / wrong_answer / format_error tracking)
uv run python -m src.scripts.train_es ... --track-decomposition
uv run python -m src.scripts.analyze_failures ...
```

A walk-through of the end-to-end pipeline (probe → train → plot) is in [`.final-deliverable/demo.ipynb`](.final-deliverable/demo.ipynb).

---

## 9. Project history — how we got here

The two source-of-truth PDFs are the **third pivot** of the project. The earlier bi-weekly reports in `.report/` are kept for context — they record an evolving research direction that is, on its face, only loosely related to the final story but in retrospect supplied the infrastructure and motivating questions that the final argument needed.

- **Week 4 — Prompt-policy PG vs ES on a Wordle environment ([`020626-week4-report.md`](.report/020626-week4-report.md)).**
  We initially defined the problem as comparing REINFORCE and ES over Bernoulli prompt policies inside Prime Intellect's hosted Wordle environment, treating the LLM as a black-box reward oracle. The setup verified that both methods could be wired against a non-differentiable, externally-evaluated reward, but the prompt-module action space was too narrow to surface interesting optimization dynamics.

- **Week 6 — Reproducing Augmented Random Search ([`022026-week6-report.md`](.report/022026-week6-report.md)).**
  We pivoted to a controlled, classical setting and reproduced Mania et al. (2018). The headline finding — that **reward normalization alone** accounts for essentially all of ARS's `~14,600×` improvement over Basic Random Search on LQR — is the same mechanism that returns at scale in Lesson 3: the *handling of advantage magnitude* is the load-bearing piece, far more than population size, top-`b` selection, or state normalization on its own. ARS scaled polynomially while vanilla ES and REINFORCE collapsed exponentially as state dimension grew (`d = 256` LQR, ~43 million× gap).

- **Week 8 — Calibration framework for ES on OPT-350M ([`030626-week8-report.md`](.report/030626-week8-report.md)).**
  We pivoted again to real LLM weights. A 4×4 (`σ`, `α`) calibration grid on OPT-350M / RTE produced the `σ = 3e-4`, `α = 1e-3` operating point and validated the seed-based in-place perturbation infrastructure that the final scripts still use.

- **Week 10 — Mechanism ablations on BoolQ ([`032726-week10-report.md`](.report/032726-week10-report.md)).**
  Five-way ablation (one-sided vs two-sided, Gaussian vs Rademacher, normalize on/off, top-k, population scaling) on OPT-350M / OPT-1.3B. The null results dominate: at fixed forward-pass budget, two-sided ≈ one-sided, Gaussian ≈ Rademacher, top-k = 1 matches top-k = N — consistent with a low-dimensional effective subspace. The rise-then-decay curve at `N = 1` and the `N = 2` collapse (initially attributed to budget allocation) flagged two questions that the final report would later answer: *why does small `N` fail*, and *what does normalization actually do at small `N`*.

- **Week 12 — First full SNR + N-cancellation theory ([`Week_12_Report.pdf`](.report/Week_12_Report.pdf)).**
  This is the long-form precursor to the workshop submission: positive degeneracy probability `q ≈ 1 / √(4π B p_0 (1−p_0)(1−ρ))`, conservative `N_min ≈ 29`, SNR-based descent argument, and the **N-cancellation theorem** at fixed budget under `η ∝ N`. The workshop paper condenses this into the cleaner "availability threshold + normalization distortion" framing used in `final-report.pdf`.

The single sentence that connects all five reports: **what looks like a tug-of-war over population size has always been a tug-of-war over reward magnitude.** ARS-on-LQR (Week 6) won by normalizing rewards across iterations; ES-on-LLMs (Week 12 → final report) wins by *not* normalizing across the population at `N = 2`. Same lever, opposite direction, because the underlying variance structure is different.

---

## 10. Limitations

- The `N = 2` normalize-off recovery is demonstrated on one model–task pair (Qwen2.5-1.5B on GSM8K and TREC), with three seeds. We do not claim `N = 2` is universally sufficient.
- The benign/malignant taxonomy (Lesson 2 in the slides) is a working conjecture validated on three (model, task) pairs; it is intentionally not in the workshop paper, which restricts its claims to the formally provable mechanisms.
- The degeneracy probability is derived under a homogeneous-batch, small-perturbation, iid-correctness assumption. Proposition 3 is an approximation; we treat `ρ` as a complementary diagnostic, and at small `B` / extreme `p_0` we recommend the empirical probe over the formula.
- The N-cancellation theorem (Week 12) assumes `η ∝ N` and a slowly-moving gradient; the workshop version restricts to the moment-level drift–diffusion statement under local linearization at fixed `θ`.
- The spectral-decay argument (Appendix H) is order-of-magnitude; the exponent `β ≈ 2` and effective rank `r ≈ 100` are imported from Liang et al. (2026) and not re-measured on our models.
- Some Week-12 experiments (OPT-13B SST-2 with CE reward; ES-at-Scale Qwen-2.5-Instruct / Countdown) remain inconclusive due to implementation gaps or multi-GPU requirements; see Week 12 §11.3.

---

## 11. References

- Malladi, Gao, Nichani, Damian, Lee, Chen, Arora. **Fine-Tuning Language Models with Just Forward Passes.** NeurIPS 2023. [arXiv:2305.17333]
- Qiu, Gan, Hayes, Liang, Xu, Dailey, Meyerson, Hodjat, Miikkulainen. **Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning.** 2026. [arXiv:2509.24372]
- Liang, Song, Liu, Gore, Fiete, Miikkulainen, Qiu. **The Blessing of Dimensionality in LLM Fine-Tuning: A Variance–Curvature Perspective.** 2026. [arXiv:2602.00170]
- Salimans, Ho, Chen, Sidor, Sutskever. **Evolution Strategies as a Scalable Alternative to Reinforcement Learning.** 2017. [arXiv:1703.03864]
- Mania, Guy, Recht. **Simple Random Search of Static Linear Policies is Competitive for Reinforcement Learning.** NeurIPS 2018.
- Nesterov, Spokoiny. **Random Gradient-Free Minimization of Convex Functions.** Foundations of Computational Mathematics, 2017.
- Sun, Shao, Qian, Huang, Qiu. **Black-Box Tuning for Language-Model-as-a-Service.** ICML 2022.
- Gao, Xu, Ye, Liu, He, Fu, Mei, Wang, Wu. **On Designing Effective RL Reward at Training Time for LLM Reasoning.** 2024. [arXiv:2410.15115]

Full reference list in `.final-deliverable/final-report.pdf`.
