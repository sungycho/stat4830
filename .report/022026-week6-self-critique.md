# Self-Critique — Week 6 Deliverable

**OODA Framework**

---

## Observe

Running the full experiment pipeline end-to-end confirms the code works: 222 JSON result files generated, 11 figures produced, two-phase protocol (grid search → eval100) completes without errors. ARS V2-t achieves −17.17±0.51 on LQR across 20 seeds. However, the Pendulum eval100 never reached the −200 threshold in any seed, and MuJoCo is not yet evaluated.

---

## Orient

### Strengths

- **Full algorithmic reproduction.** All five ARS variants (BRS, V1, V1-t, V2, V2-t) are implemented and validated against each other. The BRS→V1 ablation isolates reward normalization as the single most important component (14,600× improvement on LQR).
- **Comprehensive sensitivity sweeps.** Three hyperparameter axes (σ, α, N) are fully characterized with concrete optimal ranges (σ ∈ [0.03, 0.3], α ∈ [0.005, 0.01], N=16 is a sweet spot). This goes beyond the paper's own appendix analysis.
- **Two-phase protocol infrastructure.** Grid search → best_configs.json → eval100 pipeline works end-to-end. The code in `src/sweep_protocol.py` matches the paper's exact evaluation procedure and is reusable for MuJoCo.

### Areas for Improvement

- **Eval100 phase ran only 20 seeds, not 100.** The paper's Figure 1 and reliability claims are based on 100-seed runs. The current 20-seed run is insufficient to produce proper percentile-band plots. All "eval100" results in this report should be labeled "eval20."
- **Pendulum did not converge — grid was too coarse.** The hyperparameter grid only covered α ∈ {0.01, 0.05} and N ∈ {8, 16}, missing the wider ranges (α=0.1, N=32–64) that likely enable convergence. The −200 threshold was never reached in any of the 8 grid configurations, making the best_configs.json selection for Pendulum effectively random (all tied at maximum penalty score 3200).
- **No MuJoCo results.** The paper's headline claims (ARS matches SAC/PPO on HalfCheetah and Humanoid) require MuJoCo evaluation. LQR and Pendulum are toy problems. Without locomotion results, the central research question — "is simple random search competitive for RL?" — cannot be answered from these experiments.
- **Baseline strength is limited.** Current comparisons are against REINFORCE and Vanilla ES. We have not yet tested stronger modern alternatives (e.g., PPO and TRES), so we cannot claim ARS is competitive against current best-in-class implementations.

### Critical Risks / Assumptions

The report implicitly assumes LQR and Pendulum results are representative of the broader paper findings, but they are not — the paper's title specifically refers to locomotion tasks. The polynomial scaling advantage of ARS over REINFORCE at high dimension is an encouraging signal, but scaling in a linear system (LQR) may not transfer to the nonlinear dynamics of MuJoCo. Additionally, the Pendulum non-convergence raises a concern: if ARS struggles on a 3-dimensional nonlinear system within 3200 episodes, it may require much larger budgets on higher-dimensional MuJoCo tasks than originally estimated.

---

## Decide

### Concrete Next Actions

1. **Run eval100 with 100 seeds on LQR.** Change `--n-seeds 100` in `sweep_protocol.py`'s eval100 phase and re-run; update figures with proper percentile bands using `plot_percentile_curves()` to match Figure 1 of the paper.
2. **Fix Pendulum grid: add α ∈ {0.05, 0.1, 0.2}, N ∈ {32, 64}, budget=10000.** Re-run Phase 1 grid search for Pendulum only, then Phase 2 with winning config. Target: at least some seeds reaching −200.
3. **Install MuJoCo and run Swimmer and Hopper first.** These are the simplest locomotion tasks. Run `uv add "gymnasium[mujoco]"` then `--sweep mujoco_comparison --budget 50000` limited to just `--tasks swimmer hopper` to get at least two data points from the paper's Table 1.
4. **Upgrade baselines for fairness.** Add more recent methods such as PPO (policy-gradient baseline) and TRES (advanced ES baseline) instead of vanilla PG and ES, under matched budgets to test whether ARS gains hold against stronger methods.
5. **Bridge to project end goal (LLM fine-tuning).** Add a small pilot that studies ARS/ES behavior in a high-dimensional, non-smooth reward setting and focuses on robust hyperparameter tuning strategy first; then test more ambitious algorithmic changes. This direction is motivated by *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*.

---

## Act

### Resource Needs

The MuJoCo sweep requires a Linux machine (MuJoCo rendering is not supported on macOS without a display server in headless mode); WSL on Windows or a Linux server is needed. Each MuJoCo task at budget=50000 with N=16 directions takes approximately 12–30 CPU-hours sequentially; a 16-core machine with Ray enabled reduces this to ~2–4 hours per task. The eval100 expansion to 100 seeds adds ~45 minutes on LQR (already fast at ~10s/seed) and is feasible on the current machine.
