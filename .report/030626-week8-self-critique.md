# Self-Critique — Week 8 Deliverable

**OODA Framework**

---

## Observe

Calibration completed: $4\times4$ grid over $(\sigma, \alpha)$ on RTE, 3 seeds each. Zero-shot baseline is 0.455. Best config is $\sigma=3\times10^{-4}$, $\alpha=10^{-3}$, mean peak val_acc = 0.533 (+7.8% over zero-shot). Most configs in the safe operating range ($\sigma \leq 10^{-3}$, $\alpha \geq 10^{-3}$) achieve 0.527–0.533 (+7–8%). $\sigma=10^{-2}$ is the only regime that degrades below zero-shot (0.496). Low lr ($\alpha=10^{-4}$) produces high variance. All ablation blocks are deferred to the next deliverable using the calibrated HPs.

---

## Orient

### Strengths

- **Budget-controlled comparison.** Using training forward passes (not iterations, not wall-clock time) as the x-axis is the right choice for comparing variants that differ in per-iteration cost. One-sided ES with $2\times$ iterations consumes exactly the same training budget as two-sided, so any performance difference is attributable to the estimator design, not compute.
- **Seed-based in-place perturbation is memory-efficient.** Peak memory overhead is the size of the largest single layer, not the full model. This enables ES on models where storing explicit noise vectors (or model copies) would be infeasible.
- **Modular task registry.** Adding a third task (e.g., HellaSwag, WinoGrande) requires creating one file and one import. The same runner and plotter work without modification.
- **Three-pass antithetic trick.** Saves 25% of parameter scan time per seed vs. the four-pass naive approach, with identical mathematical result.

### Areas for Improvement

- **Scope is limited to calibration this week.** The ablation blocks — the primary scientific contribution — are deferred. The calibration result itself is only instrumentally useful (picking HPs); the comparisons that answer *why* certain choices work are what matter for the research question.
- **Calibration with one seed per config is noisy.** Running 16 configs $\times$ 1 seed is faster but the best cell in the heatmap may be a statistical outlier rather than a robust optimum. Ideally, the top 3 configs from the single-seed pass would be rerun with 3 seeds before committing to a best HP.
- **Generation scoring is fragile.** `max_new_tokens=4` with greedy decoding assumes the label token appears first. OPT-series models sometimes generate preamble (e.g., whitespace, punctuation) before the yes/no token, producing spurious $-1$ rewards. This degrades calibration accuracy independently of ES dynamics.
- **Single model family.** All experiments use OPT-350M. ES behavior may differ substantially across model families due to different implicit loss landscapes, tokenizer vocabularies, and pretraining distributions.

### Critical Risks / Assumptions

The calibration result is more encouraging than the theoretical SNR estimate suggested. A +7.8% gain over zero-shot at $N=8$ and $d\approx330M$ indicates the signal is detectable despite the $O(1/\sqrt{Nd}) \sim 10^{-5}$ bound — likely because the relevant fine-tuning directions are concentrated in a much lower-dimensional subspace than $d$ implies. The more pressing risk is the hard failure mode at $\sigma=10^{-2}$: the transition from +7% to -4% over a single order-of-magnitude increase in $\sigma$ means the safe operating range has a sharp upper boundary. If the ablation comparisons require a slightly different optimal $\sigma$ per variant (e.g., Rademacher noise may tolerate larger $\sigma$ due to bounded perturbations), the fixed calibrated HP could disadvantage some variants.

---

## Decide

The immediate technical roadmap is in the report (Section 4). One decision worth flagging here: if calibration shows no config improving over zero-shot, **diagnose before running ablations** — check (a) whether the scorer mismatch with OPT's generation format inflates $-1$ rewards, (b) whether sigma is outside the viable range, (c) whether the batch size is too small for a meaningful advantage signal. Running the full ablation suite on a broken reward function would waste all the compute budget.

---

## Act

### Future Directions (4th deliverable and beyond)

#### Systematic dissection of SOTA ZO/ES mechanisms

The recent literature has produced a cluster of specialized methods — MeZO, HiZOO, ESSA, BSZO, FZOO, Sparse MeZO — each claiming gains over vanilla ES on LLM fine-tuning. The goal is to implement these in the same controlled framework (same tasks, same forward-pass budget accounting) and answer: *which component is doing the work, and does it hold up under a fair compute comparison?*

**Curvature exploitation (MeZO, HiZOO).** MeZO implicitly aligns with the empirical Fisher by reusing the same perturbation for the gradient estimate and the weight update. HiZOO adds an explicit diagonal preconditioner. The key question is whether this curvature signal is worth its estimation cost — a preconditioner estimated from $O(1)$ ZO samples may be too noisy to help, and large $N$ may already smooth out the curvature effect through averaging.

**Subspace restriction (ESSA, LoRA tangent space).** Projecting perturbations onto a low-dimensional task-relevant subspace improves the SNR of each gradient estimate at the cost of restricting the search space. The honest comparison must charge the subspace identification (PCA or LoRA forward passes) to the budget. If the effective task-relevant dimension $k$ is small, this is a major win; if it is not, the subspace is just a bias.

**Multi-direction fusion (BSZO, FZOO).** Variance-weighted aggregation of gradient estimates across directions is theoretically sound (analogous to control variates), but each additional direction costs forward passes. The relevant question is whether the variance reduction per added direction exceeds the loss from fewer update steps within a fixed budget — a curve that likely peaks at a moderate number of directions and then reverses.

**Sparse updates (Sparse MeZO).** Restricting updates to a subset of layers reduces the parameter scan cost per step and concentrates the update signal. The risk is that the task-relevant gradient has components in the frozen layers, introducing a bias that accumulates across iterations. Whether this bias is tolerable depends on the task: classification heads may localize to output layers, while deeper reasoning tasks may require updates throughout.

#### Theory (if time permits)

**Variance scaling in high-dimensional LLM parameterizations.** The ES gradient estimator has variance $\propto d$ for Gaussian noise; at $d \sim 10^8$ this is severe. Rademacher noise has the same variance in the worst case but may behave better empirically when the reward Hessian is sparse (few large eigenvalues dominate). A formal analysis of how the fine-tuning loss landscape interacts with noise distribution choice would make the Gaussian vs. Rademacher comparison interpretable beyond the empirical outcome.

**Population size vs. effective dimension.** If the task-relevant parameter subspace has dimension $k \ll d$, then $N \sim k$ perturbations suffice to span it and larger $N$ yields diminishing returns. Estimating $k$ from the stable rank of the gradient covariance across iterations would give a principled stopping rule for the population-scaling sweep, rather than treating $N$ as a pure hyperparameter.
