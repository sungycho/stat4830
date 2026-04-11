# Self-Critique — Week 12 Deliverable

**OODA Framework**

---

## Observe

The week 12 report presents a reward-aware theory of population size for ES fine-tuning, organized into four results: (1) a closed-form degeneracy probability P(A=0) ≈ 1/√(4πBp₀(1−p₀)(1−ρ)) for binary reward, (2) a minimum population theorem N_min = ⌈log δ / log P(A=0)⌉, (3) an N-cancellation theorem showing total training progress is independent of N ∈ [N_min, r] at fixed forward-pass budget with η∝N, and (4) a sandwich bound N_min ≤ N* ≤ r unifying both the MeZO (N=1) and ES-at-scale (N≈30) results.

Empirically, we report two things: (a) a degeneracy probe on Qwen2.5-0.5B-Instruct/GSM8K at σ=0.001, K=200 pairs, at two batch sizes — B=16 (p₀=0.234, P(A=0)_emp=0.215, theory=0.235, +9% error, ρ̂≈0.40, N_min=2) and B=4 (p₀=0.250, P(A=0)_emp=0.370, theory=0.461, +25% error, ρ̂≈0.22, N_min=4); and (b) training curves showing OPT-1.3B on BoolQ flat across N∈{1,4,8,16,32} under binary reward, and OPT-13B on SST-2 flat at ≈0.62 with CE reward over 2k iterations, compared to MeZO's ≈0.92.

---

## Orient

### Strengths

- **The theoretical framework is complete and internally consistent.** The four results connect cleanly: the degeneracy formula gives P(A=0), which gives N_min, which defines the lower end of the sandwich, while Liang et al.'s effective rank r gives the upper end. The N-cancellation proof is clean and the three-step decomposition (per-step progress → substitute η=N/r → sum over T) makes the cancellation algebraically transparent.

- **The ρ parameter genuinely resolves the 2π vs 4π discrepancy.** Introducing intra-pair correlation ρ = Corr(X⁺_j, X⁻_j) as an explicit parameter bridges the independent-perturbation assumption (ρ=0, 4π formula) and the empirical behavior (ρ≈0.4, fitting closer to the 2π special case). This is a real theoretical contribution, not a patch.

- **The probe validates the formula's order of magnitude.** At B=16, the 9% theory-empirical gap is well within the CLT approximation error, and N_min=2–4 is a self-consistent prediction for a model with p₀≈0.25. The probe also correctly identifies the CLT assumption as the reason B=4 shows larger (25%) error.

- **The flat OPT-1.3B BoolQ curves are qualitatively consistent with theory.** All N∈{1,4,8,16,32} produce identical flat curves — precisely what the theory predicts when p₀ is low and N_min is not satisfied for any tested N. This is not proof, but it is a genuine consistency check.

### Areas for Improvement

- **The N_min=29 conservative bound is not clearly scoped.** The report presents N_min=29 as matching the empirical N≈30 without prominently flagging that this requires p₀<0.1, i.e., a near-random uninstructed base model. Our own probe shows N_min=2–4 for Qwen2.5-0.5B-Instruct (p₀≈0.25). The two regimes — uninstructed base models vs. capable instruction-tuned models — lead to N_min differing by an order of magnitude. This distinction should be the lead finding in the empirical section, not a footnote.

- **The SST-2 result is an implementation bug, not a limitation.** The report describes the 0.62 flat curve as a "limitation" (Section 11.3, item 3), but it is a known implementation discrepancy — restricted-vocabulary softmax vs. MeZO's full-vocabulary CE. Framing a bug as a limitation understates the problem and leaves the experiment's status ambiguous. The experiment is uninformative until the CE formulation is fixed.

- **ρ was back-calculated rather than directly measured.** The probe report acknowledges that ρ̂ was computed from the empirical P(A=0) via formula (7) rather than directly from stored (R⁺_k, R⁻_k) pairs. This is circular: the formula used to derive ρ̂ is the same formula whose accuracy ρ̂ is supposed to improve. The code was updated after the probe to store raw pairs, but the reported table uses the back-calculated value.

- **N∝η scaling is assumed but never verified experimentally.** The N-cancellation theorem requires η∝N for exact cancellation, and Proposition 4 shows that at fixed η, smaller N strictly wins. Our training experiments use fixed η. The report does not state which scaling was used in any of the training runs, making it impossible to know which regime the flat curves reflect.

### Critical Risks / Assumptions

- **The central empirical claim is untested.** The hypothesis — training fails below N_min and succeeds above it — has no direct experimental support. The flat OPT-1.3B curves show failure, but they are for a model where N_min might require p₀<0.1, which we never probe. A single well-controlled experiment (probe to measure N_min on a specific model/task, then train at N=1, N=N_min, N=2N_min, N=30) would either confirm or falsify the theory. Without it, the theory is plausible but unvalidated.

- **ρ is measured at one (model, task, σ) point.** The report uses ρ̂≈0.40 from Qwen2.5-0.5B/GSM8K/σ=0.001 as if it characterizes a general regime. Whether ρ is stable across models, tasks, and σ values is completely unknown. If ρ varies widely, the general formula's practical utility depends on running the probe every time, making N_min a measurement rather than a prediction.

- **The "inverted-U" peak at N≈30 is partially an artifact of fixed η** (Remark 4, Section 5.2). Our previous population-scaling experiments (Weeks 8–10) used fixed η. The apparent N*≈30 optimum in those experiments may reflect step-count collapse at large N rather than a genuine degeneracy threshold. This reinterpretation is correct and important, but it retroactively undermines the earlier framing of N*≈30 as evidence for the degeneracy theory.

---

## Decide

Three concrete decisions:

1. **Run a threshold experiment on a model where N_min is measurable.** First probe a model on a task where p₀ is high enough (e.g., Qwen2.5-1.5B-Instruct on BoolQ) to get N_min in the 2–8 range. Then train at N ∈ {1, 2, N_min, 2·N_min, 30} under matched forward-pass budget with η∝N. This is the minimum experiment needed to validate the theory.

2. **Fix the CE implementation before running more SST-2/OPT-13B experiments.** Switch from restricted-vocabulary softmax to full-vocabulary softmax (2-line change in `hf_backend.py`), verify the loss matches MeZO's reported formulation, then rerun at 2k iterations. Only after confirming parity with MeZO at short runs should the full 20k-iteration experiment be submitted.

3. **Compute ρ̂ directly in the next probe run.** The updated `probe_degeneracy.py` already stores (R⁺_k, R⁻_k) pairs. The next probe should report both the directly-computed ρ̂ and the back-calculated one, confirming they agree — or quantifying the discrepancy.

---

## Act

### Pending experiments

```bash
# 1. Probe Qwen2.5-1.5B-Instruct on BoolQ to measure N_min
uv run python -m src.scripts.probe_degeneracy \
  --task boolq --model Qwen/Qwen2.5-1.5B-Instruct \
  --sigma 0.001 --K 200 --batch-size 16

# 2. Fix CE and run short SST-2 parity check
# Edit src/backends/hf_backend.py: full-vocab softmax
# Then:
uv run python -m src.scripts.train_es \
  --task sst2 --model facebook/opt-13b \
  --reward ce --n 1 --iters 500 --save

# 3. Threshold experiment (after step 1 gives N_min)
# Run N = 1, 2, N_min, 2*N_min, 30 under matched budget with eta = eta_0 * N
```

### Research directions opened by this week

- **Dynamic N_min scheduling.** Since p₀ changes during training, N_min follows a U-shape (high at init, low mid-training, high near convergence). An adaptive strategy that starts at N=N_min(init), decreases N as p₀ rises, then switches to CE reward near convergence could reduce compute cost by 2–5× compared to fixed N=30.

- **ρ vs. σ characterization.** Running the probe at σ ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} on one model/task would directly test whether ρ→1 as σ→0 and ρ→0 as σ→∞, as the theory predicts. This would take one GPU-hour and would either validate the formula's σ-dependence or reveal a constant-ρ regime.

- **N-cancellation direct test.** Run two matched training experiments — one with η∝N (N∈{2,4,8,16} all with matched total budget) and one with fixed η — and plot final accuracy vs. N. The η∝N run should produce a flat line; the fixed-η run should decrease monotonically with N. This would be the cleanest possible empirical demonstration of the theorem.
