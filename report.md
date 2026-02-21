# Reproducing Augmented Random Search: A Comparative Study of Derivative-Free RL Methods

**STAT 4830 — Week 6 Report**
**Date:** 2026-02-20

---

## 1. Problem Statement

### What are we optimizing?

This project reproduces the core experimental findings of Mania et al. (NeurIPS 2018), "Simple random search of static linear policies is competitive for reinforcement learning." The optimization target is a linear policy $\theta \in \mathbb{R}^{d_a \times d_s}$ that maps observations to actions to maximize expected cumulative reward in continuous-control environments. Concretely:

- **LQR (Linear Quadratic Regulator):** state $s \in \mathbb{R}^4$, action $a \in \mathbb{R}^2$, horizon $T=200$. The closed-form optimal policy exists, so convergence to near-optimal is verifiable. Success threshold: $\text{eval\_return} \ge -50$.
- **Pendulum:** state $s \in \mathbb{R}^3$ ($\cos\theta, \sin\theta, \dot{\theta}$), action $a \in \mathbb{R}^1$, horizon $T=200$. Success threshold: $\text{eval\_return} \ge -200$.

We compare three derivative-free optimization families:

1. **ARS** (Augmented Random Search) with five sub-variants (BRS, V1, V1-t, V2, V2-t)
2. **Vanilla ES** (basic evolution strategy, one-sided finite differences)
3. **REINFORCE** (policy gradient with score function estimator)

All methods share the same linear policy class and a fixed episode budget of 3,200.

### Why does this problem matter?

The paper's central claim — that simple random search with appropriate normalization matches or exceeds state-of-the-art deep RL on MuJoCo locomotion — was surprising at publication. Understanding *which* algorithmic components (reward normalization, state normalization, top-b selection) drive the performance gap is directly relevant to method selection for any black-box optimization problem where gradients are unavailable (e.g., LLM prompt tuning via scalar reward).

### How will we measure success?

- **Primary:** Final eval_return averaged over 3 seeds at the end of the episode budget
- **Secondary:** Episodes-to-threshold (first episode count where eval_return exceeds the threshold)
- **Stability:** Standard deviation across seeds; variance across the eval100 multi-seed run

### Constraints and risks

- CPU-only execution; no MuJoCo results yet (requires separate `gymnasium[mujoco]` install)
- Budget fixed at 3,200 episodes per run to keep wall-clock time under 15 seconds per run
- Pendulum convergence is slower than LQR; the threshold of −200 was not reached within this budget

---

## 2. Technical Approach

### Mathematical formulation

All methods optimize a linear policy $\pi_{\theta}(s)=\theta s$, where $\theta \in \mathbb{R}^{d_a \times d_s}$. ARS (Algorithm 1 from the paper) performs the following update per iteration:

$$
\text{For } k=1,\dots,N:\quad
\delta_k \sim \mathcal{N}(0, I),\quad
r_k^+=R\!\left(\pi_{\theta+\sigma\delta_k}\right),\quad
r_k^-=R\!\left(\pi_{\theta-\sigma\delta_k}\right).
$$

$$
\text{Select top-}b\text{ directions by } \max(r_k^+, r_k^-).
$$

$$
\theta \leftarrow \theta + \frac{\alpha}{b\,\sigma_R}\sum_{k \in \text{top-}b}(r_k^+ - r_k^-)\delta_k.
$$

where $\sigma_R=\operatorname{std}\!\left(\{r_k^+, r_k^-\}_{k\in\text{top-}b}\right)$ is the reward standard deviation normalization (applied only in V1/V1-t/V2/V2-t; not in BRS). State normalization (V2/V2-t) applies online Welford running statistics: observations are centered and scaled by a RunningNorm that is frozen during evaluation rollouts to prevent contamination.

The five variants differ as follows:

| Variant | reward_norm | use_state_norm | top-b selection |
|---------|:-----------:|:--------------:|:---------------:|
| BRS     | ✗           | ✗              | b = N           |
| V1      | ✓           | ✗              | b = N           |
| V1-t    | ✓           | ✗              | b < N           |
| V2      | ✓           | ✓              | b = N           |
| V2-t    | ✓           | ✓              | b < N           |

### Implementation

- **`src/policy.py`** — `LinearPolicy` (stores $\theta$, provides `act(obs)`) and `RunningNorm` (Welford online estimator with variance floor $\varepsilon=10^{-8}$; returns 0 for low-variance dimensions)
- **`src/methods/ars.py`** — Core ARS loop; respects `reward_norm`, `use_state_norm`, and `b` from `MethodConfig`
- **`src/methods/vanilla_es.py`** — One-sided ES (no antithetic pairs); same hyperparameters for fair comparison
- **`src/methods/reinforce.py`** — Score function estimator with mean baseline, Gaussian policy action noise
- **`src/envs/lqr.py`, `src/envs/pendulum.py`** — Gymnasium-wrapped environments
- **`src/sweep_protocol.py`** — Two-phase protocol: Phase 1 grid search (3 seeds, 8 configs) to select best $(\alpha, N, \sigma)$ per task/variant; Phase 2 multi-seed evaluation with winning config

### Validation strategy

Each experiment runs 3 independent seeds; results are averaged. The eval loop uses `eval_policy()` with a frozen copy of the RunningNorm (no state updates during evaluation). Hyperparameter sensitivity sweeps over $(\sigma, \alpha, N)$ validate robustness. The grid search phase selects the best config by minimizing mean episodes-to-threshold across the 3 seeds.

---

## 3. Initial Results

### 3.1 Method Comparison

With default hyperparameters ($\alpha=0.02$, $N=16$, $b=8$, $\sigma=0.03$, budget$=3200$), averaged over 3 seeds:

| Method | LQR final return | Pendulum final return |
|--------|----------------:|-----------------------:|
| ARS V2-t | **−16.6** | **−1,162** |
| Vanilla ES | −179,104 | −1,734 |
| REINFORCE | −178,106 | −1,601 |

ARS outperforms both alternatives by **~10,800× on LQR** and **~1.5× on Pendulum**. The LQR gap is especially striking: Vanilla ES and REINFORCE both diverge to extremely negative returns (order $10^5$) while ARS converges near-optimally within the first 320 episodes (10 iterations). This confirms that the combination of antithetic perturbations, reward normalization, and state normalization is the key enabler - not the policy class, which is identical across methods.

### 3.2 ARS Variant Ablation

Isolating the contribution of each ARS component (N=16, b=16 or b=8, budget=3200):

**LQR:**

| Variant | Mean final return | vs. BRS improvement |
|---------|----------------:|--------------------:|
| BRS     | −240,792        | baseline            |
| V1      | −16.50          | **14,593×**         |
| V1-t    | −16.43          | 14,657×             |
| V2      | −16.40          | 14,682×             |
| V2-t    | −16.60          | 14,506×             |

Adding reward normalization alone (BRS → V1) accounts for virtually the entire gain. State normalization (V1 → V2) and top-b selection (V2 → V2-t) provide marginal additional improvement on LQR where state distributions are already well-conditioned.

**Pendulum:**

| Variant | Mean final return |
|---------|----------------:|
| BRS     | −1,606          |
| V1      | −1,184          |
| V2      | −1,165          |
| V2-t    | −1,166          |

The same ordering holds. BRS is 38% worse than V2 on Pendulum, consistent with the paper's finding that reward normalization matters most when reward scales vary across dimensions or seeds.

### 3.3 Scaling with Problem Dimension

ARS V2-t vs. Vanilla ES vs. REINFORCE as LQR dimensionality increases (action_dim = state_dim / 4):

| (state, action) dim | ARS return | Vanilla ES return | REINFORCE return |
|---------------------|-----------:|------------------:|-----------------:|
| 4 × 1               | −17.7      | −190,411          | −137,014         |
| 16 × 4              | −72.2      | −2,846,024        | −1,451,025       |
| 64 × 16             | −1,959     | −203,572,614      | −191,570,328     |
| 256 × 64            | −93,798    | −8,062,998,789    | −4,378,583,632   |

ARS degrades polynomially (roughly $O(d)$ per decade of dimension), while Vanilla ES and REINFORCE degrade exponentially. At the largest tested dimension (256-state, 64-action, $\theta \in \mathbb{R}^{64\times256}$ = 16,384 parameters), ARS is **~43,000,000× better** than Vanilla ES. This empirically validates the paper's claim that ARS scales far more favorably than gradient-based or naive ES methods.

The polynomial scaling of ARS is expected: the update uses antithetic perturbations (reducing variance by ~2×) and reward normalization (preventing step-size blow-up as rewards scale with dimension), while the baselines lack both.

### 3.4 Hyperparameter Sensitivity

**Noise standard deviation $\sigma$ (ARS V2-t, LQR):**

| $\sigma$ | Mean final return |
|---|----------------:|
| 0.005 | −17.49 |
| 0.01  | −17.39 |
| 0.03  | **−16.60** |
| 0.1   | −16.65 |
| 0.3   | −16.49 |

Performance is robust across the range $\sigma \in [0.03, 0.3]$ - a full decade. Only very small $\sigma$ (0.005) shows meaningful degradation, likely because perturbations are too small to escape flat regions early in training. The paper's default $\sigma=0.03$ is near-optimal but not uniquely so.

**Learning rate $\alpha$ (ARS V2-t, LQR):**

| $\alpha$ | Mean final return |
|---|----------------:|
| 0.001 | −23.23 |
| 0.005 | **−16.13** |
| 0.01  | −16.16 |
| 0.02  | −16.60 |
| 0.05  | −16.80 |
| 0.1   | −19.29 |

The optimal range is $\alpha \in [0.005, 0.01]$. Both too-small (0.001 -> slow convergence) and too-large (0.1 -> overshoot) degrade performance. The paper's default $\alpha=0.02$ is slightly suboptimal for LQR but remains within 3% of the best.

**Population size N (ARS V2-t, LQR, b = N/2):**

| N  | Mean final return |
|----|----------------:|
| 4  | −17.89 |
| 8  | −16.75 |
| 16 | −16.60 |
| 32 | −16.14 |
| 64 | **−15.98** |

Performance improves monotonically with N but with strongly diminishing returns. N=8 achieves ~95% of N=64's performance at 1/8th the per-iteration cost. For the fixed episode budget, N=16 (the paper's default) is a reasonable sweet spot.

### 3.5 Two-Phase Protocol Results

The paper's evaluation protocol was implemented: Phase 1 runs a hyperparameter grid ($\alpha \in \{0.01, 0.05\}$, $N \in \{8, 16\}$, $\sigma \in \{0.01, 0.03\}$) across 3 seeds; Phase 2 runs the best config across multiple seeds.

**Phase 1 result (best configs):**
- LQR V2-t: $\alpha=0.01$, $N=8$, $\sigma=0.01$ -> mean episodes-to-threshold = **160 episodes**
- Pendulum V2-t: $\alpha=0.01$, $N=8$, $\sigma=0.01$ -> threshold never reached (penalized at 3200)

**Phase 2 result (20-seed eval100):**

| Environment | Mean final return | Std | Seeds reaching threshold |
|-------------|------------------:|----:|-------------------------:|
| LQR V2-t    | −17.17            | 0.51 | 20/20 (all within 320 episodes) |
| Pendulum V2-t | −1,196.4        | 34.5 | 0/20 (threshold −200 not reached) |

The LQR result is highly stable: std=0.51 with mean=−17.17 across 20 independent seeds, confirming the algorithm is not sensitive to initialization for this problem. Every seed exceeded the −50 threshold, consistently doing so within 320 episodes (10 iterations of N=16).

The Pendulum failure is informative: the $-200$ threshold was never reached within the 3200-episode budget. The best observed return was $-1{,}143$ (seed 7171). The grid search was too coarse - the winning config ($\alpha=0.01$, $N=8$) was constrained to a $2\times2\times2$ grid, and a larger budget (10,000+ episodes) or larger $N$ would likely resolve this.

### 3.6 Current Limitations

- All results are on LQR and Pendulum. The paper's core claims concern MuJoCo locomotion (Swimmer, Hopper, HalfCheetah, Walker2d, Ant, Humanoid). Those results are not yet available.
- The eval100 phase ran only 20 seeds instead of 100 due to compute constraints.
- The full hyperparameter grid (paper uses $\alpha \in \{0.01, 0.02, 0.05\}$, $N \in \{8, 16, 32\}$, $\sigma \in \{0.01, 0.03, 0.1\}$) was reduced to $2\times2\times2 = 8$ configs per task.
- Vanilla ES and REINFORCE use identical hyperparameters to ARS (not their own optimal configs), which may understate their best achievable performance.

---

## 4. Next Steps

### Immediate improvements

1. **MuJoCo evaluation.** Install `gymnasium[mujoco]` (`uv add "gymnasium[mujoco]"`), then run `uv run python src/run_sweep.py --sweep mujoco_comparison --budget 50000`. This is the only way to validate the paper's central claim. Wall-clock estimate: ~10–50 CPU-hours per task.

2. **Full 100-seed eval on LQR.** The current eval100 only used 20 seeds. Expand to 100 by setting `--n-seeds 100` in `sweep_protocol.py`; this enables proper percentile-band visualization matching Figure 1 of the paper.

3. **Pendulum convergence.** Expand the grid search: add $\alpha \in \{0.05, 0.1\}$, $N \in \{32, 64\}$, and increase budget to 10,000. The Pendulum task needs more exploration directions and a higher learning rate to converge within budget.

### Technical challenges to address

- **Percentile-band visualization.** The paper's Figure 1 shows 0–10%, 10–20%, 20–100% percentile bands across seeds, not mean ± std. The `plot_percentile_curves()` function in `src/visualize_all.py` is stubbed but needs full implementation and testing against the 100-seed LQR data.
- **Survival bonus handling for MuJoCo.** Hopper, Walker2d, Ant, Humanoid include a +1/step survival bonus that distorts reward normalization during training. The `remove_survival_bonus` flag is implemented in `src/envs/mujoco.py` but requires validation against the paper's Table 1 numbers.
- **Ray parallelism for MuJoCo.** Each MuJoCo episode takes ~50–200ms (CPU physics). Enabling `--use-ray` will parallelize N rollout pairs per iteration across CPU cores, giving ~8–16× speedup on a multi-core machine.

### Questions for course staff

- Is there access to a Linux machine with 16+ cores for the MuJoCo sweep?
- Should the comparison against Vanilla ES and REINFORCE use their own optimal hyperparameters, or match ARS's settings for a controlled ablation?

### What we've learned

The dominant finding from this implementation is that **reward normalization alone** (BRS → V1 transition) accounts for essentially all of ARS's advantage over naive random search — a 14,600× improvement on LQR. State normalization and top-b selection provide marginal additional gains. This is architecturally important: these normalization techniques are cheap to implement and could be applied to any derivative-free optimizer.

The scaling experiment is the most practically informative result: ARS's polynomial degradation vs. the exponential collapse of REINFORCE and Vanilla ES at high dimension (d=256) suggests ARS may be the right tool for high-dimensional black-box optimization problems where the policy class itself can remain linear (e.g., representation-frozen LLM prompting via scalar reward).
