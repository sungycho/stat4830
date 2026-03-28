# Self-Critique — Week 10 Deliverable

**OODA Framework**

---

## Observe

Five mechanism ablations completed on OPT-350M / BoolQ (5 seeds each): one-sided vs. two-sided, Gaussian vs. Rademacher noise, normalization on/off, population scaling N∈{4,8,16,32}, and top-k at N=16. All ablations produced null results within ±0.005 of each other. Three extended experiments this week: top-k at N=8 (corrected), top-k at N=64 (5 k-values), and population scaling on OPT-1.3B across N∈{1,2,4,8,16,32,64,128}.

Key numbers: zero-shot baseline = 0.094. Top-k null result holds at both N=8 and N=64. N=1 on OPT-1.3B shows clear rise-then-decay peaking near 0.622 then decaying. N=8 reaches 0.621 stably. N=2 collapsed to 0.297 due to normalization confound. N=64/128 failed due to only 4 and 2 gradient steps in the fixed budget.

---

## Orient

### Strengths

- **Rise-then-decay cleanly replicated.** The N=1 OPT-1.3B learning curve shows a textbook non-monotonic trajectory — the first clean empirical demonstration of the stiff/flat mode competition mechanism from Liang et al. (2025) in a real LLM fine-tuning setting with scalar rewards.

- **Top-k null result is theoretically grounded.** All k values — including k=1 — produce identical final accuracy at both N=8 and N=64. This is interpretable through the degeneracy lens: the fine-tuning landscape improvement subspace is low-dimensional, so filtering perturbations by advantage magnitude adds no signal. The result holds across an 8× range of population sizes, ruling out N-specific explanations.

- **Normalization confound caught and corrected.** z-score over k≤2 samples always outputs ±1 regardless of advantage scale, destroying the gradient signal. The corrected top-k=2 experiment (normalization off) produces curves identical to top-k=1, confirming the confound was the sole cause of the earlier spurious underperformance.

- **Fixed compute budget is strictly maintained.** Every comparison — across N, across k, across model size — uses the same 7,680 training forward passes, so differences are attributable to algorithm design rather than resource differences.

### Areas for Improvement

- **N=2 confound was not caught before the OPT-1.3B run.** The normalization issue for k=2 was discovered while analyzing top-k results, but the pop_scaling_1b block had already been submitted with normalization ON for N=2. A pre-run design checklist — "does any variant have population_size or top_k ≤ 2 with normalize=True?" — would have caught this before spending GPU compute.

- **N=64/128 budget exhaustion was predictable from the formula.** With $\text{iters}(N) = \lceil 7680 / (N \times 32) \rceil$, N=64 gives 4 steps and N=128 gives 2. This was computable before running. The experiment still consumed disk and GPU time to confirm something the math already implied. Either simulate iteration counts before submitting or explicitly note degenerate-budget variants as informational-only.

- **Disk failure mid-run at N=64/128 in the first attempt.** Model checkpoint files filled remote storage, causing a `RuntimeError: iostream error`. Required a full second run on a fresh disk. Default flag for large-N runs should disable checkpoint saving.

- **The top-k=2 confounded result was published in the first version of this plot before the issue was caught.** The initial N=8 top-k figure included the misleading top-k=2 curve, which temporarily misled the analysis. Any k≤2 or N≤2 variant with normalization enabled should be treated as invalid at design time.

- **N=4 underperformance on OPT-1.3B is unexplained.** N=4 has 60 iterations (sufficient budget), yet peaks at only 0.460 — well below N=1 (0.622) and N=8 (0.621). This is surprising and currently unexplained. One hypothesis is that N=4 sits in an unstable regime where gradient noise is too high for OPT-1.3B but the per-seed budget is too low to average it out. This warrants investigation before drawing conclusions about the N=4–8 range.

### Critical Risks / Assumptions

- **"Best peak" metric masks decay.** N=1 achieves mean peak 0.622, equal to N=8, but the N=1 curve decays after that peak. If the budget were 2× larger, N=1 would end significantly below N=8. Reporting only peak accuracy is misleading for variants with non-monotonic curves. Final-iteration accuracy should be reported alongside peak for a complete picture.

- **All results are on BoolQ with OPT-series models.** Whether the top-k null result, the rise-then-decay at N=1, and the N=8 optimality generalize to other tasks (e.g., reasoning benchmarks) or other model families (LLaMA, Mistral) is unknown. The degeneracy interpretation may be BoolQ-specific if the task's relevant parameter subspace is unusually simple.

---

## Decide

Three concrete decisions follow from this week:

1. **Rerun N=2 on OPT-1.3B with `--no-normalize`.** The current N=2 data point is invalid and must be replaced before any scaling law fitting. This is the highest-priority pending experiment.

2. **Retire all k≤2 / N≤2 variants with normalization ON.** Add a validation check in the experiment runner that raises an error — not a warning — when this condition is met.

3. **Report both peak and final-iteration accuracy** for all non-monotonic variants (N=1 and any future variants showing decay).

---

## Act

### Pending experiments

```bash
# Fix N=2 on OPT-1.3B
uv run python -m src.scripts.run_experiment_week10 --block pop_scaling_1b_n2_fix --device cuda
```

### Research directions opened by this week

- **Compute-optimal N\* (Chinchilla-style).** The N=1–32 OPT-1.3B curves give enough signal to fit a power law $N^*(B) \propto B^\alpha$. This is a directly publishable empirical finding — analogous to Chinchilla scaling for ES hyperparameter scheduling.

- **ES vs. PG landscape comparison.** The rise-then-decay in N=1 is the cleanest geometric signal obtained so far. Running REINFORCE under matched compute tests whether the same trajectory appears with gradient-based optimization — directly addressing the original PG vs. ES framing and grounding both methods in the same landscape theory.

- **Curvature-active rank estimation.** The top-k null result implies the improvement subspace has low effective rank $d_\text{eff}$. Logging the per-iteration gradient covariance across the N=8 population and tracking its spectral decay would directly measure $d_\text{eff}$ — the quantity that unifies the top-k null result, the rise-then-decay, and the N=8 optimality into a single geometric picture.
