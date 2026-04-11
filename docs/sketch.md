# Sketch: Gradient Norm, Smoothed Gradient, and the Blowup

---

## What is the gradient norm?

$\nabla f(\theta)$ is a vector in $\mathbb{R}^d$ — one partial derivative per parameter. It points in the direction of steepest ascent of $f$.

$\|\nabla f(\theta)\|$ is just the **length** of that vector. Large norm = steep landscape, far from a stationary point. Small norm = near-flat region or minimum.

Why do we care about $\|\nabla f\|^2$ specifically? Because that's how much progress SGD makes per step:

$$f(\theta - \eta \nabla f) \approx f(\theta) - \eta \|\nabla f\|^2$$

So $\|\nabla f\|^2$ is the "fuel" available at each step. If it's zero, you're stuck regardless of learning rate.

---

## Why ES estimates the smoothed gradient, not the true gradient

Start with the ES estimator:

$$\hat{g} = \frac{1}{\sigma} \mathbb{E}_\varepsilon\!\left[f(\theta + \sigma\varepsilon)\,\varepsilon\right], \quad \varepsilon \sim \mathcal{N}(0, I)$$

Take its expectation using **Stein's lemma**: for $\varepsilon \sim \mathcal{N}(0, I)$,

$$\mathbb{E}[h(\varepsilon)\,\varepsilon] = \mathbb{E}[\nabla_\varepsilon h(\varepsilon)]$$

Apply this with $h(\varepsilon) = f(\theta + \sigma\varepsilon)$:

$$\mathbb{E}[f(\theta + \sigma\varepsilon)\,\varepsilon] = \mathbb{E}[\nabla_\varepsilon f(\theta + \sigma\varepsilon)] = \sigma\,\mathbb{E}[\nabla_\theta f(\theta + \sigma\varepsilon)]$$

So:

$$\mathbb{E}[\hat{g}] = \mathbb{E}_\varepsilon[\nabla_\theta f(\theta + \sigma\varepsilon)] = \nabla_\theta \underbrace{\mathbb{E}_\varepsilon[f(\theta + \sigma\varepsilon)]}_{f_\sigma(\theta)}$$

ES is unbiased for $\nabla f_\sigma(\theta)$ — the gradient of the **Gaussian-blurred version** of $f$, not $f$ itself.

**Intuitively**: $f_\sigma(\theta)$ is $f$ convolved with a Gaussian of width $\sigma$. It's a blurry copy of $f$ — sharp corners are rounded, local minima are filled in. ES always estimates where this blurred function is going, not the original.

---

## Why does bias to the true gradient exist?

$$\nabla f_\sigma(\theta) - \nabla f(\theta) = \mathbb{E}_\varepsilon[\nabla f(\theta + \sigma\varepsilon)] - \nabla f(\theta)$$

Taylor-expand $\nabla f(\theta + \sigma\varepsilon)$ around $\theta$:

$$\nabla f(\theta + \sigma\varepsilon) = \nabla f(\theta) + \sigma \nabla^2 f(\theta)\,\varepsilon + \frac{\sigma^2}{2}\nabla^3 f(\theta)[\varepsilon, \varepsilon] + \cdots$$

Take expectation — odd powers of $\varepsilon$ vanish ($\mathbb{E}[\varepsilon] = 0$), even powers survive ($\mathbb{E}[\varepsilon_i \varepsilon_j] = \delta_{ij}$):

$$\mathbb{E}_\varepsilon[\nabla f(\theta + \sigma\varepsilon)] = \nabla f(\theta) + \frac{\sigma^2}{2} \underbrace{\nabla^3 f(\theta) \cdot \mathrm{tr}(I)}_{\text{third-order curvature} \times d} + O(\sigma^4)$$

So the **bias scales as $O(\sigma^2 \cdot d)$**. The smoothed gradient points in a slightly wrong direction because you're averaging gradients over a ball of radius $\sim\sigma$ — and the ball gets bigger in proportion to $d$ (the Gaussian spreads out in all $d$ directions).

This gives the $\sigma$ trade-off:
- **Small $\sigma$**: bias $\to 0$ ✓, but variance $\propto 1/\sigma^2$ explodes ✗
- **Large $\sigma$**: variance is small ✓, but gradient points toward the wrong optimum ✗

---

## Connecting to the blowup: why does $\|\hat{g}\|^2$ inflate by $(d+n-1)/n$?

The estimator $\hat{g}$ has two components mixed together:

$$\hat{g} = \underbrace{\nabla f_\sigma(\theta)}_{\text{true signal}} + \underbrace{\text{noise from random } \varepsilon}$$

The noise term comes from sampling $\varepsilon \sim \mathcal{N}(0, I_d)$ — a $d$-dimensional random vector. Even if $f$ is completely flat (zero gradient), $\hat{g}$ has expected squared norm:

$$\mathbb{E}[\|\hat{g}\|^2] \approx \frac{d}{n} \cdot \frac{\mathrm{Var}[f]}{\sigma^2}$$

The $d$ comes from the fact that $\|\varepsilon\|^2 = d$ in expectation — your random perturbation spreads across all $d$ dimensions, so the noise in $\hat{g}$ has $d$ independent components adding up.

MeZO Lemma 2 makes this exact:

$$\mathbb{E}\!\left[\|\hat{\nabla} L\|^2\right] = \frac{d + n - 1}{n} \cdot \mathbb{E}\!\left[\|\nabla L\|^2\right]$$

The optimizer **cannot tell signal from noise** — it just sees $\|\hat{g}\|^2$ and assumes that's $\|\nabla f\|^2$. So it thinks there's $(d+n-1)/n$ times more gradient fuel than there really is, and takes steps that are proportionally too large $\to$ diverges unless you shrink $\eta$ by $n/(d+n-1)$.

**The whole chain in one sentence**: ES mixes a true gradient signal with $d$-dimensional isotropic noise, which inflates the apparent gradient norm by a factor of $d$, which forces the learning rate to shrink by $1/d$, which is why ES is so slow — except that in practice the effective dimension is $r \ll d$, and that's what saves it.

---

## Antithetic sampling: what actually changes

The estimator written above is one-sided. The antithetic version subtracts the negative perturbation:

$$\hat{g} = \frac{1}{2N\sigma} \sum_{i=1}^{N} \left[f(\theta + \sigma\varepsilon_i) - f(\theta - \sigma\varepsilon_i)\right] \varepsilon_i$$

This still estimates $\nabla f_\sigma(\theta)$ — still unbiased for the smoothed gradient. The Stein's lemma argument goes through identically:

$$\mathbb{E}\!\left[\frac{f(\theta+\sigma\varepsilon) - f(\theta-\sigma\varepsilon)}{2\sigma}\,\varepsilon\right] = \mathbb{E}[\nabla_\theta f(\theta + \sigma\varepsilon)] = \nabla f_\sigma(\theta)$$

**But antithetic sampling reduces variance.** Why? The term $f(\theta - \sigma\varepsilon)$ is negatively correlated with $f(\theta + \sigma\varepsilon)$ — when one is high the other tends to be low. Subtracting them amplifies the signal (the difference grows with the gradient) while canceling the baseline noise (both sides share the same $f(\theta)$ baseline):

$$\mathrm{Var}\!\left[f(\theta+\sigma\varepsilon) - f(\theta-\sigma\varepsilon)\right] < \mathrm{Var}\!\left[f(\theta+\sigma\varepsilon) - f(\theta)\right]$$

So the MeZO blowup factor $(d+n-1)/n$ still holds — but with antithetic sampling each pair counts as one effective sample, giving twice the variance reduction per forward pass compared to one-sided ES. That's why antithetic pairs are the default in practice.

---

## Why is $\|\varepsilon\|^2 = d$ in expectation?

Because $\varepsilon \sim \mathcal{N}(0, I_d)$ means each coordinate is independently drawn as $\varepsilon_i \sim \mathcal{N}(0, 1)$.

The squared norm is:

$$\|\varepsilon\|^2 = \varepsilon_1^2 + \varepsilon_2^2 + \cdots + \varepsilon_d^2$$

Take the expectation of each term:

$$\mathbb{E}[\varepsilon_i^2] = \mathrm{Var}[\varepsilon_i] + (\mathbb{E}[\varepsilon_i])^2 = 1 + 0 = 1$$

So:

$$\mathbb{E}[\|\varepsilon\|^2] = \sum_{i=1}^{d} \mathbb{E}[\varepsilon_i^2] = d \cdot 1 = d$$

Intuitively: each of the $d$ dimensions contributes exactly 1 unit of spread on average, and they add up linearly. This is also why $\|\varepsilon\| \approx \sqrt{d}$ — the vector has unit variance per coordinate but $d$ coordinates, so its length concentrates around $\sqrt{d}$ in high dimensions.

---

## Why is gradient noise $\propto d/N$, and can we replace $d$ with effective rank $r$?

**Deriving the $d/N$ scaling for L-smooth $f$**

For a single antithetic pair, write $c(\varepsilon) = \frac{f(\theta+\sigma\varepsilon) - f(\theta-\sigma\varepsilon)}{2\sigma}$. By Taylor expansion to leading order (valid because $f$ is L-smooth):

$$c(\varepsilon) \approx \varepsilon^\top \nabla f(\theta)$$

So $\|\hat{g}\|^2 \approx c(\varepsilon)^2 \|\varepsilon\|^2 = (\varepsilon^\top \nabla f)^2 \|\varepsilon\|^2$. Take the expectation using Gaussian moment formulas ($\mathbb{E}[\varepsilon_i^4]=3$, $\mathbb{E}[\varepsilon_i^2\varepsilon_j^2]=1$ for $i\neq j$, odd moments $= 0$):

$$\mathbb{E}\!\left[(\varepsilon^\top\nabla f)^2\|\varepsilon\|^2\right] = \sum_{i,j,k} \nabla f_i \nabla f_j\, \mathbb{E}[\varepsilon_i\varepsilon_j\varepsilon_k^2] = (d+2)\|\nabla f\|^2$$

So for $N$ samples:

$$\mathbb{E}[\|\hat{g}\|^2] \approx \frac{d+2}{N}\|\nabla f\|^2$$

The $d$ factor is exact — it comes directly from $\|\varepsilon\|^2 = d$ spreading noise across all $d$ dimensions simultaneously.

**Replacing $d$ with effective rank $r$**

The gradient $\nabla f(\theta)$ only has significant components in a low-dimensional subspace. If the Hessian $H = \nabla^2 f(\theta)$ has $r \ll d$ large eigenvalues and the rest are near zero, then $\nabla f$ approximately lives in that $r$-dimensional subspace:

$$\nabla f(\theta) \approx P_r\, \nabla f(\theta)$$

where $P_r$ projects onto the top-$r$ eigenvectors of $H$. Decompose $\hat{g}$ into signal and noise:

$$\hat{g} = \underbrace{(\varepsilon^\top\nabla f)\,P_r\varepsilon}_{\text{signal-aligned}} + \underbrace{(\varepsilon^\top\nabla f)\,(I - P_r)\varepsilon}_{\text{wasted noise in } (d-r)\text{-dim null space}}$$

The wasted noise contributes to $\|\hat{g}\|^2$ but produces zero useful update. If we only care about progress along $\nabla f$, the effective noise inflates by $r$, not $d$:

$$\mathbb{E}[\|\hat{g}\|^2]_{\text{effective}} \approx \frac{r+2}{N}\|\nabla f\|^2$$

This gives corrected versions of the MeZO results:

$$\eta_{\text{ZO}} = \frac{n}{r + n - 1} \cdot \eta_{\text{SGD}} \qquad \text{(much less conservative than } n/(d+n-1)\text{)}$$

$$T = O\!\left(\frac{r}{n} \cdot \text{landscape terms}\right) \qquad \text{(scales with } r\text{, not } d\text{)}$$

And crucially: $N^* \approx r$. Once $n \geq r$, extra samples reduce within-subspace variance but the gain is negligible — you're averaging over more directions than the gradient subspace has.

**Is this a reasonable assumption?**

Yes, strongly supported:
- **Aghajanyan et al. 2021** ("Intrinsic Dimensionality Explains...") empirically showed LLM fine-tuning has intrinsic dimension $r \sim O(100)$–$O(1000)$, far below $d \sim 350\text{M}$.
- **LoRA** works for the same reason — the gradient update is approximately low-rank.
- **MeZO Lemma 3** proves convergence with $r$ not $d$ under "local $r$-effective rank of the Hessian" (their Assumption 1), validated empirically.

The caveat: $r$ is not fixed — it depends on $\theta$ and decreases as you approach convergence. This is consistent with our observation that ES improves fast early then stalls: as $r$ shrinks, flat modes become inaccessible.

---

## Does an optimal $N^*$ exist? Bottlenecks and partial results

**The fundamental obstacle: $N$ cancels at fixed budget**

For L-smooth non-convex $f$, the standard SGD-with-noise convergence bound is:

$$\frac{1}{T}\sum_t \mathbb{E}[\|\nabla f(\theta_t)\|^2] \leq O\!\left(\sqrt{\frac{L\,\sigma_g^2}{T}}\right)$$

where $\sigma_g^2 \propto d/N$. At fixed budget $B = N \times T$, substitute $T = B/N$:

$$\text{error} \propto \sqrt{\frac{L \cdot d/N}{B/N}} = \sqrt{\frac{Ld}{B}}$$

The $N$'s cancel perfectly. In standard smooth non-convex theory, $N$ does not appear in the convergence rate — only $B$ matters. Any proof of $N^*$ must go beyond this framework.

**Route 1: Discrete reward degeneracy (lower bound on $N$)**

With binary reward $F \in \{-1, +1\}$ and $P(F=+1) = p$, the probability that all $N$ samples return the same reward — making $\mathrm{std}(F_1,\ldots,F_N) = 0$ and the gradient update degenerate — is:

$$P_{\text{degen}}(N) = p^N + (1-p)^N$$

The effective convergence rate is reduced by $(1 - P_{\text{degen}}(N))$:

$$\text{effective error} \propto \frac{\sqrt{Ld/B}}{1 - p^N - (1-p)^N}$$

This blows up as $N \to 1, 2$. This gives a clean **lower bound**:

$$N \geq N_{\min} = O\!\left(\frac{\log(1/\delta)}{\log(1/\max(p,\, 1-p))}\right)$$

for $\delta$-probability of non-degenerate updates. Directly motivated by our BoolQ results ($p \approx 0.62$).

**Route 2: Efficiency saturation at effective rank $r$ (upper bound on $N$)**

Once $N > r$, additional samples explore directions outside the gradient subspace — costing forward passes but contributing zero useful signal. Define:

$$\text{efficiency}(N) = \frac{\text{gradient signal quality}}{\text{forward passes per step}} \propto \frac{\min(N,\, r)}{N}$$

This is constant for $N \leq r$ and strictly $\propto 1/N$ for $N > r$. Efficiency is maximized at $N = r$, giving an **upper bound** $N \leq O(r)$.

**Route 3: Non-convex landscape — $T$ matters non-linearly (hard)**

In non-convex settings, you need at least $T_{\min}$ steps to escape local minima. If $N$ is too large, $T = B/N < T_{\min}$ and the optimizer never escapes. This gives $N \leq B/T_{\min}$, but $T_{\min}$ is landscape-dependent and not easily characterized — this route is the least formalizable.

**Combining Routes 1 and 2:**

$$\underbrace{O\!\left(\frac{\log(1/\delta)}{\log(1/\max(p,\,1-p))}\right)}_{N_{\min},\ \text{discrete degeneracy}} \leq N^* \leq \underbrace{O(r)}_{N_{\max},\ \text{efficiency saturation}}$$

Both bounds arise from first principles — one from reward structure, one from landscape geometry.

**The honest bottleneck:** The smooth theory gives no interior optimum within this interval — the $N$-cancellation means we cannot yet pin down exactly where $N^*$ sits. Closing this requires either a tighter non-convex landscape assumption or empirical fitting: sweep $N$ at fixed $B$ across tasks with different $r$, measure $N^*$, and check whether $N^* \approx r$. That empirical validation is a legitimate contribution even without a closed-form proof.

---

## Binary vs Cross-Entropy Reward: How They Work and Why It Matters

### How CE works vs binary, mechanically

```
Binary:  prompt → generate() → decode text → regex match → ±1
         cost: max_new_tokens forward passes per example

CE:      prompt + label → forward() → log P(label tokens | prompt)
         cost: 1 forward pass per example
```

For CE, you tokenize the prompt and the correct label together, run one forward pass, and read off the log-probability at the label token positions. No generation, no decoding, no regex. That is where the 2–15× wall-clock speedup comes from.

### Why not just use CE despite the speedup — four real reasons

**Reason 1 — CE is a proxy that can diverge from accuracy.** CE optimizes $\log P(\text{correct label})$, which can increase without accuracy changing. If the model raises the correct class logit from 0.30 to 0.35 but the argmax was already correct at 0.30, CE improves and accuracy stays flat. The reverse also happens: accuracy can improve while CE worsens. For a research study comparing ES variants, using CE introduces a confound — you cannot tell if a variant is better at maximizing accuracy or just better at maximizing log-probability.

**Reason 2 — CE does not apply to generation tasks.** For Countdown, GSM8K, math500 — there is no single correct token sequence. The model might output "42", "forty-two", or "the answer is 42". CE requires knowing exactly which tokens to score against, which is undefined for free-form generation. Binary outcome reward is the only sensible choice for these tasks.

**Reason 3 — CE requires white-box access.** CE requires the model's logit distribution over the vocabulary. API-based models expose only decoded text — no logits. Binary reward is universally applicable; CE is white-box only.

**Reason 4 — Chat template complexity.** Instruct models (Qwen, Llama-3-Instruct) wrap prompts in a chat template. To extract $\log P(\text{label} \mid \text{prompt})$ correctly, you need to apply the template to the prompt but then score the label as a bare continuation — not wrapped in another assistant turn. This is subtle to implement and can introduce tokenization artifacts.

### Does the right reward type depend on the task?

Yes — the task structure constrains which reward types are even available:

| Task type | Binary ±1 | Cross-entropy | Contrastive margin |
|---|---|---|---|
| Binary classification (SST-2, BoolQ) | ✓ natural | ✓ works | ✓ works |
| Multi-class classification (TREC, SST-5) | ✓ works | ✓ works, faster | ✓ works, most principled |
| Extractive QA (SQuAD, ReCoRD) | ✓ (F1 threshold) | ✗ no fixed label tokens | ✗ no runner-up defined |
| Math / reasoning (GSM8K, Countdown) | ✓ only option | ✗ no fixed token sequence | ✗ no runner-up defined |
| Free-form generation | ✓ (BLEU / reward model) | ✗ | ✗ |

The proxy gap between CE and accuracy grows with task complexity. For binary tasks, there is only one decision boundary — raising $\log P(\text{correct})$ almost always eventually crosses it. For 6-class TREC, CE can redistribute probability among all 5 wrong classes in ways that never change the argmax. For math, there is no relationship between $\log P(\text{"42"})$ and whether the reasoning chain is correct.

**Binary reward is always aligned with the eval metric. CE is aligned only approximately, and less so as $K$ grows.**

### Is $N$ truly indifferent within the sandwich bound $[N_\text{min}, r]$?

The answer is: **indifferent in convergence rate, but not in accuracy ceiling.**

From the N-cancellation (smooth theory): within the plateau where $p_\text{eff}(N) \approx 1$, total progress at fixed budget $B$ is:

$$\text{Progress}(N) \approx \frac{B}{d}(\|\nabla f\|^2 - 1)$$

$N$ drops out — rate is the same for all $N \in [N_\text{min}, r]$.

But from the stiff/flat mode analysis (arXiv 2602.00170), the terminal plateau is:

$$1 - J_\infty(N) = \frac{C}{N}, \quad C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2 - \alpha\lambda_i}$$

Larger $N$ gives a strictly better accuracy ceiling, even within the sandwich bound. Combining both:

$$J(B, N) \approx \left(1 - \frac{C}{N}\right)\!\left(1 - e^{-B/(N\tau)}\right) + J_0\,e^{-B/(N\tau)}$$

Three sub-regimes within the sandwich bound depending on budget $B$ relative to convergence timescale $\tau \sim r/\alpha$:

| Budget regime | Condition | N-indifferent? |
|---|---|---|
| Very small budget | $B \ll N\tau$ | Yes — far from plateau, rate dominates, $N$ cancels |
| Moderate budget | $B \sim N\tau$ | No — $N^* = 2C$ is the genuine optimum |
| Large budget | $B \gg N\tau$ | No — ceiling dominates, larger $N$ wins, push toward $N = r$ |

**Practical reading**: being outside the sandwich bound causes large discontinuous drops in performance (degeneracy below $N_\text{min}$, too few steps above $r$). Being inside the sandwich bound, the difference between any two choices of $N$ is governed by the ceiling term $C/N$ — small at moderate budgets, which is why population scaling plots look flat across $N \in \{4, 8, 16, 32\}$. You are reading off the rate-dominated regime, not the ceiling-dominated regime.

### The key insight the field missed

ES decouples the training signal from the evaluation metric — the reward is just a scalar fed into the gradient estimator. This means the choice of reward function is a **free design variable** that gradient-based methods never had.

- Gradient methods must use CE (need $\nabla L$, CE is differentiable)
- ES can use any scalar: binary accuracy, CE, contrastive margin, process reward, anything

The field has not recognized this because ES papers either inherit CE from gradient methods without questioning it (MeZO), or come from RL where the environment provides the reward (OpenAI ES, ARS). The question "what is the optimal training signal for ES?" has never been asked.

The degeneracy framework gives the first principled answer:
- Binary: $\delta > 0$, $N_\text{min} > 0$, directly optimizes accuracy, degrades with $K$
- CE: $\delta = 0$, $N_\text{min} = 0$, log-prob proxy, $K$-invariant convergence
- Contrastive margin $\log P(\text{correct}) - \log P(\text{runner-up})$: $\delta = 0$, $N_\text{min} = 0$, boundary-focused, closer to accuracy than CE

The optimal reward is not CE by default. It depends on $K$, $N$, $B$, and white-box access. Deriving this from first principles — and validating it empirically — is a genuine open contribution.

---

---


## Stiff vs Flat Modes: Alpha, Lambda, Terminal Ceiling, and C — Detailed Derivation

### What alpha and lambda-i are

**α** = the ES learning rate. Maps directly to `--lr` in the code. Controls how large a weight update you make after each iteration.

**λᵢ** = the i-th eigenvalue of $\mathbf{H} = -\nabla^2 J(\theta^*)$, the negative Hessian of the reward function at the optimum $\theta^*$. Each eigenvalue measures curvature along its corresponding eigenvector direction in weight space:

- Large $\lambda_i$ → reward changes rapidly along that direction → **stiff mode**
- Small $\lambda_i \approx 0$ → reward barely changes along that direction → **flat mode**

These are properties of the reward landscape, not the model architecture. They are task-dependent — different $\lambda_i$ for SST-2 vs TREC on the same model.

### What J-infinity(N) means

$J_\infty(N)$ is the **asymptotic accuracy ceiling** that ES converges to as $T \to \infty$, as a function of population size $N$.

It is not the theoretical maximum accuracy. It is the ceiling imposed by the noise floor: because ES continuously injects perturbations of scale $\sigma$ at every step, it never fully settles at $\theta^*$ — it oscillates around it in a steady state. The size of that oscillation determines the gap $1 - J_\infty(N)$. Even at infinite budget, ES with finite $\sigma$ and finite $N$ leaves residual error. Larger $N$ reduces this residual; smaller $\sigma$ also reduces it but slows convergence.

### The per-mode OU derivation

Near $\theta^*$, expand the reward quadratically:

$$J(\theta^* + x) \approx J(\theta^*) - \frac{1}{2}x^\top \mathbf{H} x, \quad \mathbf{H} = -\nabla^2 J \geq 0$$

Diagonalize $\mathbf{H}$ — write $x$ in the eigenbasis so each coordinate $x_i$ corresponds to eigenvalue $\lambda_i$. The ES dynamics decouple per mode (arXiv 2602.00170, Eq. 7):

$$x_{i,t+1} = \underbrace{(1 - \alpha\lambda_i)}_{\text{decay factor}}\,x_{i,t} + \underbrace{\frac{\alpha\sigma}{\sqrt{N}}\,\xi_{i,t}}_{\text{injected noise}}$$

This is an **Ornstein-Uhlenbeck process** per mode — deterministic decay toward zero plus Gaussian noise injected each step. The stationary variance solves $v_{i,\infty} = (1-\alpha\lambda_i)^2 v_{i,\infty} + (\alpha\sigma)^2/N$:

$$v_{i,\infty} = \frac{\alpha\sigma^2}{N\lambda_i(2 - \alpha\lambda_i)}$$

**Stiff mode** (large $\lambda_i$):
- Decay factor $(1-\alpha\lambda_i)$ is small → fast pull back toward $\theta^*$ each step
- Stationary variance $v_{i,\infty}$ is small → model stays close to optimum
- Convergence timescale $\tau_i \sim (\alpha\lambda_i)^{-1}$ → short

**Flat mode** (small $\lambda_i \approx 0$):
- Decay factor $(1-\alpha\lambda_i) \approx 1$ → almost no restoring force each step
- Stationary variance $v_{i,\infty} \approx \alpha\sigma^2 / (2N\lambda_i)$ → blows up as $\lambda_i \to 0$
- Convergence timescale $\tau_i \sim (\alpha\lambda_i)^{-1}$ → extremely long

The terminal reward gap is curvature × stationary variance summed over all modes:

$$1 - J_\infty = \frac{1}{2}\sum_i \lambda_i\,v_{i,\infty} = \frac{\alpha\sigma^2}{2N}\sum_{\lambda_i > 0} \frac{1}{2 - \alpha\lambda_i}$$

### What C is and what it looks like numerically

$$C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2 - \alpha\lambda_i}, \quad \text{so} \quad 1 - J_\infty(N) = \frac{C}{N}$$

In the regime where $\alpha\lambda_i \ll 1$ (holds for most modes since $\alpha$ is small), $\frac{1}{2-\alpha\lambda_i} \approx \frac{1}{2}$, so:

$$C \approx \frac{\alpha\sigma^2}{2} \cdot \frac{r}{2} = \frac{\alpha\sigma^2 r}{4}$$

**Typical values** ($\alpha = 10^{-4}$, $\sigma = 10^{-3}$, $r = 100$):

$$C \approx \frac{10^{-4} \times 10^{-6} \times 100}{4} = 2.5 \times 10^{-9}$$

This gives $1 - J_\infty(N=8) \approx 3 \times 10^{-10}$ — essentially zero. That seems to predict near-perfect accuracy, which contradicts observation.

**Why the numbers don't match directly**: the formula is derived in the paper's normalized coordinate system near $\theta^*$. The actual $\lambda_i$ values are unobservable without computing the Hessian. A handful of near-boundary stiff modes (where $\alpha\lambda_i \to 2$) can make $\frac{1}{2-\alpha\lambda_i} \gg \frac{1}{2}$ and dominate the sum, pushing $C$ orders of magnitude above the naive estimate.

**What C is useful for in practice — relative scaling:**

$$C \propto \alpha \cdot \sigma^2 \cdot \sum_i \frac{1}{2-\alpha\lambda_i}$$

- Halve $\sigma$ → $C$ drops 4× → plateau gap halves at same $N$
- Double $N$ → $1 - J_\infty$ halves → better ceiling
- Increase $\alpha$ → $C$ grows and stability margin $2 - \alpha\lambda_i$ shrinks → worse plateau

The absolute value of $C$ requires knowing the actual spectrum. The relative scaling is what matters for choosing $(\alpha, \sigma, N)$.

### Is the J(B, N) formula from the paper or ours?

**Ours** — the paper does not have it.

arXiv 2602.00170 gives the per-mode OU dynamics (Eq. 7), the stationary variance (Eq. 17), and the terminal plateau $1 - J_\infty$ (Eq. 19). What they do not analyze is **finite-budget behavior** — they study $T \to \infty$, not fixed budget $B = NT$.

We derived $J(B, N)$ by combining three pieces:

1. **Their** terminal plateau: $J_\infty(N) = 1 - C/N$
2. **Standard** exponential approach to a fixed point: for an OU process with timescale $\tau \sim r/\alpha$ (from MeZO Theorem 1), the gap to steady state decays as $e^{-T/\tau}$:
$$J(T) \approx J_\infty - (J_\infty - J_0)\,e^{-T/\tau}$$
3. **Our** budget substitution $T = B/N$, giving:
$$J(B, N) \approx \left(1 - \frac{C}{N}\right)\!\left(1 - e^{-B/(N\tau)}\right) + J_0\,e^{-B/(N\tau)}$$

Differentiating with respect to $N$ and setting to zero in the small-budget regime ($B \ll N\tau$) gives $N^* = 2C$ — fully our derivation, synthesizing the paper's steady-state result with the finite-budget constraint from Section 3. The original paper never considers this trade-off.

---

## The Scale Mismatch in N* = 2C, and Deriving N* from r

### The scale mismatch is genuine

The $N^* = 2C$ formula came from differentiating $J(B,N)$ in the small-budget regime. The full correct derivation (including the dropped $(1-J_0)$ factor) gives:

$$N^* = \frac{2C}{1 - J_0}$$

With $C \approx \alpha\sigma^2 r/4 \approx 10^{-9}$ and $(1-J_0) \approx 0.83$:

$$N^* \approx \frac{2 \times 2.5 \times 10^{-9}}{0.83} \approx 6 \times 10^{-9}$$

That is not a valid integer population size. The formula produces a nonsensical number when you plug in raw hyperparameters from the code.

### Why the scale is wrong

The formula $1 - J_\infty = C/N$ is derived in the paper's normalized coordinate system near $\theta^*$. In that system, the eigenvectors of $C = -\nabla^2 J$ are rescaled so the problem is approximately isotropic, and the $\lambda_i$ values are $O(1)$ — not the $\sim 10^{-8}$ or smaller values in raw parameter-space units. When we computed $C \approx \alpha\sigma^2 r/4$ using raw $\alpha = 10^{-4}$ and $\sigma = 10^{-3}$, we mixed two different coordinate systems. For $C/N$ to equal a meaningful accuracy fraction like $0.05$, you'd need the sum $\sum_i 1/(2-\alpha\lambda_i) \sim 10^8$, which is only possible if some stiff modes have $\alpha\lambda_i \to 2$ (stability boundary). Those $\lambda_i$ values are unobservable without computing the Hessian.

**What $C$ is useful for — relative scaling only:**
- Halve $\sigma$ → $C$ drops $4\times$ → plateau gap halves at same $N$
- Double $N$ → $1 - J_\infty$ halves → better ceiling
- The absolute value of $C$ requires empirical calibration from plateau data, not raw hyperparameters

### Deriving N* from r (apples-to-apples)

Since $N$ and $r$ are both dimensionless integers, expressing $N^*$ in terms of $r$ fixes the scale problem.

**Step 1 — Effective steps from MeZO blowup (replacing $d$ with $r$)**

Gradient quality per step with $N$ seeds:

$$Q(N) = \frac{N}{r + N - 1}$$

Effective steps at budget $B = NT$:

$$T_\text{eff} = \frac{B}{N} \cdot \frac{N}{r+N-1} = \frac{B}{r+N-1}$$

This is decreasing in $N$ — quality gains from more seeds saturate at $N = r$.

**Step 2 — Normalize the plateau constant**

Define $\gamma = C/r$ (dimensionless, $O(1)$ in the paper's normalized coordinates). Then:

$$1 - J_\infty(N) = \frac{\gamma r}{N}$$

Now $\gamma$, $r$, and $N$ are all dimensionless.

**Step 3 — Optimize J(B, N) in the small-budget regime**

$$J(B, N) \approx J_0 + (1-J_0) \cdot \frac{B}{(r+N-1)\tau_0} \cdot \left(1 - \frac{\gamma r}{N}\right)$$

Set $\frac{d}{dN}\!\left[\frac{1 - \gamma r/N}{r+N-1}\right] = 0$:

$$\frac{\gamma r}{N^2}(r+N-1) = 1 - \frac{\gamma r}{N}$$

$$\gamma r(r + 2N - 1) = N^2$$

**Step 4 — Solve the quadratic**

$$N^2 - 2\gamma r N - \gamma r(r-1) = 0$$

$$N^* = \gamma r + \sqrt{\gamma^2 r^2 + \gamma r(r-1)} \approx r(\gamma + \sqrt{\gamma}) \approx r\sqrt{\gamma} \quad (\text{for } \gamma \ll 1)$$

**Reading the result:**

| $\gamma$ | $N^*$ | Interpretation |
|---|---|---|
| $\gamma \ll 1$ | $\approx r\sqrt{\gamma} \ll r$ | Ceiling barely matters, push $N$ small |
| $\gamma \sim 0.01$ | $\approx r/10$ | $N^* \sim 10$ if $r=100$ ✓ |
| $\gamma \sim 0.1$ | $\approx r/3$ | $N^* \sim 33$ if $r=100$ ✓ |
| $\gamma \sim 1$ | $\approx r$ | Upper bound saturated |

$N^*$ always scales linearly with $r$. The prefactor $\sqrt{\gamma}$ is a dimensionless number between 0 and 1 set by the task's reward landscape — not by raw $\alpha$ or $\sigma$.

**The full sandwich bound in consistent units:**

$$\underbrace{N_\text{min}}_{\text{degeneracy}} \leq N^* = r(\gamma + \sqrt{\gamma}) \leq r \quad \text{(all dimensionless integers)}$$

**Why this explains the empirical flatness:** for typical $\gamma \sim 0.01$–$0.1$, $N^* \in [r/10,\, r/3]$. If $r \sim 100$, that is $N^* \in [10, 33]$. Running $N \in \{8, 16, 32\}$ puts you squarely in the plateau of the $N$-cancellation regime, so performance differences are small — exactly what population scaling plots show.

**How to estimate $\gamma$ empirically:** run a pop scaling sweep, fit the observed terminal accuracy vs $N$ curve to $J_\infty(N) = 1 - \gamma r/N$. With an independent estimate of $r$ (e.g., from LoRA rank experiments or intrinsic dimensionality probes), $\gamma$ can be recovered. Then $N^* = r(\gamma + \sqrt{\gamma})$ is a quantitative prediction.

---

## The Role of Batch Size B: Variance, Unbiasedness, Convergence, and Degeneracy

### What batch size does mechanically

For each perturbation seed $i$, the reward is averaged over $B$ examples:

$$r_{\pm,i} = \frac{1}{B}\sum_{j=1}^{B} \text{score}(\text{generate}(\theta \pm \sigma\varepsilon_i,\, x_j),\, y_j)$$

The advantage: $\text{adv}_i = r_{+,i} - r_{-,i}$. In the code, the **same batch is reused for all $N$ seeds** within one iteration — this is intentional and matters (see below).

### Effect on variance

Individual reward $\text{score} \in \{-1,+1\}$ has $\text{Var}[\text{score}] = 1-(2p_0-1)^2$. Averaging over $B$:

$$\text{Var}[r_{\pm,i}] = \frac{\text{Var}[\text{score}]}{B}$$

Since antithetic perturbations are negatively correlated near decision boundaries ($+\varepsilon$ gets it right when $-\varepsilon$ gets it wrong), the advantage variance benefits from this cancellation. For the full estimator:

$$\text{Var}[\hat{g}] \propto \frac{d}{N \cdot B}$$

$B$ and $N$ are **symmetric from a variance perspective** — both reduce noise as $1/(NB)$.

### Unbiasedness

The batch reward is an unbiased estimate of $\mathbb{E}_x[\text{score}(\theta\pm\sigma\varepsilon_i, x)]$ by the law of large numbers. Since the batch and perturbation direction $\varepsilon_i$ are sampled independently:

$$\mathbb{E}[\hat{g}] = \mathbb{E}_\varepsilon\!\left[\frac{\mathbb{E}_x[\text{score}(\theta+\sigma\varepsilon,x)] - \mathbb{E}_x[\text{score}(\theta-\sigma\varepsilon,x)]}{\sigma}\cdot\varepsilon\right] = \nabla f_\sigma(\theta)$$

**Batch averaging introduces no bias** — only additional variance. The estimator remains unbiased for $\nabla f_\sigma(\theta)$ at any $B \geq 1$.

### Effect on convergence rate: the full cancellation

From variance scaling $\text{Var}[\hat{g}] \propto d/(NB)$, the standard convergence bound is:

$$\text{error after } T \text{ steps} \approx O\!\left(\sqrt{\frac{Ld}{NBT}}\right)$$

At fixed total forward-pass budget $\text{Budget} = 2NBT$:

$$\text{error} \approx O\!\left(\sqrt{\frac{2Ld}{\text{Budget}}}\right)$$

**$N$, $B$, and $T$ all cancel simultaneously.** The N-cancellation from Section 3 extends to $B$: from pure convergence rate theory, the split between $N$, $B$, and $T$ is completely indeterminate at fixed total budget. This is a stronger form of the indifference result.

### Where B and N are NOT symmetric: degeneracy

**Seed-level degeneracy** (single seed $\text{adv}_i = 0$): probability $\delta(B) \approx 1/\sqrt{\pi B p_0(1-p_0)/2}$, decreasing in $B$.

**Full-iteration degeneracy** (all $N$ seeds degenerate): $P_\text{degen} = \delta(B)^N$.

At fixed per-iteration budget $2NB$, for TREC ($p_0 = 0.17$):

| $N$ | $B$ | $\delta(B)$ | $P_\text{degen} = \delta^N$ |
|-----|-----|-------------|------------------------------|
| 4 | 32 | 0.37 | 1.9% |
| 8 | 16 | 0.53 | 0.39% |
| 16 | 8 | 0.75 | 1.0% |

$N=8, B=16$ minimizes full-iteration degeneracy here. The exponential in $N$ makes increasing $N$ more effective than increasing $B$ for reducing whole-iteration degeneracy, up to a point. The optimal split minimizes $\delta(B)^N$ subject to $2NB = \text{const}$. Substituting $B = \text{Budget}/(2N)$ and differentiating:

$$N^*_\text{degen} \propto \text{Budget} \cdot p_0(1-p_0)$$

The optimal $N$ for degeneracy grows linearly with the budget and task difficulty — and is independent of $r$.

### Effect on terminal plateau

The terminal plateau $1 - J_\infty(N) = C/N$ contains **$N$ but not $B$**. The ceiling is determined by how many perturbation directions you average — more directions = tighter oscillation around $\theta^*$. Batch size controls how accurately you evaluate each direction but does not change where you ultimately converge.

**At infinite budget: want $N$ large (for ceiling), $B$ at minimum viable value (to free budget for more seeds or steps).**

### The same-batch subtlety

**Good**: cross-seed comparisons cancel batch-level noise. If example $x_j$ is easy for all seeds, this cancels in the z-score normalization — making the normalization more meaningful.

**Bad**: if the entire batch is uninformative (all examples trivially correct or wrong for every perturbation), all $N$ seeds are simultaneously degenerate regardless of $N$. This is **batch-level degeneracy** — it does not decrease with $N$, only with $B$. Minimum viable $B$ to avoid this:

$$B_\text{min} \sim \frac{1}{4p_0(1-p_0)}$$

| Task | $p_0$ | $B_\text{min}$ |
|------|-------|----------------|
| SST-2 | 0.50 | $\sim 4$ |
| SST-5 | 0.20 | $\sim 6$ |
| TREC | 0.17 | $\sim 7$ |

### Complete picture: optimal (N, B) split at fixed budget

| Effect | Favors large $B$ | Favors large $N$ |
|--------|-----------------|-----------------|
| Gradient variance | Symmetric ($1/NB$) | Symmetric ($1/NB$) |
| Seed-level degeneracy | ✓ reduces $\delta(B)$ | — |
| Iteration-level degeneracy | — | ✓ reduces $\delta^N$ exponentially |
| Terminal plateau ceiling | ✗ doesn't appear | ✓ $J_\infty(N)$ improves |
| Batch-level degeneracy | ✓ reduces it | ✗ doesn't help |

**Optimal strategy at fixed budget**: maximize $N$ up to $\min(r,\, N^*_\text{degen})$, use minimum viable $B$ above $B_\text{min}$, let $T$ absorb the remaining budget. $B$ has a floor but no ceiling benefit beyond degeneracy avoidance. $N$ has both a floor (degeneracy) and a ceiling (effective rank $r$) with a genuine optimum in between.

The updated sandwich bound incorporating $B$:

$$N_\text{min}(\delta(B),\, N) \leq N^* \leq r, \qquad B \geq B_\text{min} \sim \frac{1}{4p_0(1-p_0)}$$

Both bounds are task-dependent through $p_0$, and $N^*$ scales with $r$ as derived in the previous section.

---

## What is C exactly, and does gamma = alpha sigma squared over four hold?

### What C is exactly

$$C = \frac{\alpha\sigma^2}{2}\sum_{\lambda_i > 0} \frac{1}{2 - \alpha\lambda_i}$$

where $\alpha$ = learning rate, $\sigma$ = perturbation scale, $\lambda_i$ = eigenvalues of $-\nabla^2 J(\theta^*)$ (reward curvature at the optimum). The terminal plateau is $1 - J_\infty(N) = C/N$ (arXiv 2602.00170, Eq. 19).

### The approximation C ≈ alpha sigma squared r / 4 and why it fails

The approximation comes from assuming $\alpha\lambda_i \ll 1$ for all modes, giving each term $\frac{1}{2-\alpha\lambda_i} \approx \frac{1}{2}$:

$$C \approx \frac{\alpha\sigma^2}{2} \cdot \frac{r}{2} = \frac{\alpha\sigma^2 r}{4} \implies \gamma = \frac{C}{r} \approx \frac{\alpha\sigma^2}{4}$$

With $\alpha = 10^{-4}$, $\sigma = 10^{-3}$: $\gamma \approx 2.5\times10^{-11}$, giving $N^* = r\sqrt{\gamma} \approx 0.0005$ — nonsensical again.

**The approximation fails because stiff modes near the stability boundary dominate $C$:**

| $\alpha\lambda_i$ | $\frac{1}{2-\alpha\lambda_i}$ | Ratio to approximation $\frac{1}{2}$ |
|---|---|---|
| 0.001 (flat) | 0.500 | $1\times$ |
| 0.5 | 0.667 | $1.3\times$ |
| 1.5 | 2.0 | $4\times$ |
| 1.9 | 10.0 | $20\times$ |
| 1.99 | 100.0 | $200\times$ |

A single stiff mode with $\alpha\lambda_i = 1.9$ contributes as much to $C$ as 20 flat modes combined. The approximation treats all modes as flat — it is only valid when no stiff modes are near the stability boundary, which is precisely the regime where the stiff/flat analysis has nothing interesting to say.

For $C$ to produce a realistic 5\% plateau gap at $N=8$ (i.e., $C = 0.4$):

$$5\times10^{-8} \times \sum_i \frac{1}{2-\alpha\lambda_i} = 0.4 \implies \sum = 8\times10^6$$

For $r=100$ modes to produce this, each would need $\alpha\lambda_i \approx 1.999987$ — every mode at the stability boundary, which is unrealistic. The honest conclusion: the realistic $C$ is driven by a small number of near-boundary stiff modes, not by $r$ flat modes each contributing $\frac{1}{2}$.

### The honest status of gamma = C/r

**$\gamma \approx \alpha\sigma^2/4$ is wrong in practice.** The real $\gamma$ is dominated by however many near-boundary stiff modes the task-model pair has. It is not computable from raw $(\alpha, \sigma, r)$ — it requires knowing the actual $\lambda_i$ spectrum.

What $\gamma$ tells you **qualitatively** (robust regardless of exact values):
- $\gamma \propto \alpha\sigma^2$ — scaling with hyperparameters is correct even if absolute value is not
- Larger $\gamma$ → more stiff modes near boundary → larger $N^*$ needed
- $\gamma$ can only be **estimated empirically** by fitting observed $J_\infty(N)$ vs $N$ from a pop scaling sweep

### What is salvageable

The formula $N^* = r(\gamma + \sqrt{\gamma})$ is the right **structural** result:

1. $N^*$ scales linearly with $r$ ✓ — testable and robust
2. The prefactor $\sqrt{\gamma}$ is an empirically-determined O(1) quantity ✓
3. $\gamma$ cannot be computed from raw hyperparameters ✗ — must be fit from plateau data

The most honest version of the claim:

> $N^* = O(r)$, with the exact prefactor determined by the effective curvature spectrum of the task-specific reward landscape near the optimum — not analytically tractable from hyperparameters alone, but estimable by fitting terminal accuracy vs $N$ from a population scaling sweep.

---

## Notation collision: two objects both called C

There is a naming collision that causes confusion throughout these notes.

**C the matrix** (from arXiv 2602.00170, the quadratic expansion near $\theta^*$):

$$J(\theta^* + x) \approx J(\theta^*) - \frac{1}{2}x^\top \mathbf{H} x, \qquad \mathbf{H} = -\nabla^2 J(\theta^*) \geq 0$$

This is a $d\times d$ matrix — the negative Hessian of the reward at the optimum. Its eigenvalues $\lambda_i$ measure curvature per mode. The paper calls this matrix $C$; we have renamed it $\mathbf{H}$ in these notes to avoid collision.

**C the scalar** (the terminal plateau constant we introduced):

$$C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2 - \alpha\lambda_i}$$

This is a scalar compressing the terminal plateau formula $1 - J_\infty(N) = C/N$. The $\lambda_i$ in this formula are the eigenvalues of the matrix $\mathbf{H}$ above.

**How they relate:**

$$\underbrace{\mathbf{H}}_{\text{matrix, }d\times d} \xrightarrow{\text{eigendecompose}} \lambda_i \xrightarrow{\text{OU stationary variance}} \underbrace{C}_{\text{scalar plateau constant}}$$

**Notation convention going forward:**

| Symbol | Type | Meaning |
|--------|------|---------|
| $\mathbf{H}$ | $d\times d$ matrix | $-\nabla^2 J(\theta^*)$, negative Hessian at optimum |
| $\lambda_i$ | scalars | eigenvalues of $\mathbf{H}$; stiff = large, flat = small |
| $C$ | scalar | $\frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2-\alpha\lambda_i}$, terminal plateau constant |
| $\gamma$ | scalar | $C/r$, dimensionless plateau constant |

---

## Deriving N* with r = 100 fixed: the stability margin picture

### Setup: uniform stiff eigenvalues

Assume $r = 100$ stiff modes all with eigenvalue $\lambda$, flat modes negligible. Define the **stability margin** $\varepsilon = 2 - \alpha\lambda \in (0,2)$. Then:

$$C = \frac{\alpha\sigma^2}{2}\cdot\frac{r}{\varepsilon}, \qquad \gamma = \frac{C}{r} = \frac{\alpha\sigma^2}{2\varepsilon}$$

$$N^* = r\sqrt{\gamma} = \frac{r\sqrt{\alpha\sigma^2}}{\sqrt{2\varepsilon}}$$

Rearranging: the stability margin required to produce a given $N^*$:

$$\varepsilon = \frac{\alpha\sigma^2 r^2}{2N^{*2}}$$

### Plugging in r = 100, alpha = 1e-4, sigma = 1e-3

| $N^*$ | Required $\varepsilon = 2-\alpha\lambda$ | Required $\alpha\lambda$ |
|--------|------------------------------------------|--------------------------|
| 1 | $5\times10^{-7}$ | $\approx 1.9999995$ |
| 4 | $3\times10^{-8}$ | $\approx 1.99999997$ |
| 8 | $8\times10^{-9}$ | $\approx 2 - 10^{-8}$ |
| 16 | $2\times10^{-9}$ | $\approx 2 - 10^{-9}$ |

For any empirically sensible $N^*$ with our hyperparameters, the dominant eigenvalue must be essentially at the stability boundary. If you observe $N^* \approx 8$ empirically, the dominant stiff mode has $\alpha\lambda \approx 2 - 10^{-8}$.

### The resolution: gamma << 1 means N is genuinely indifferent

With $\varepsilon \geq 0.01$ (any realistic margin): $\sqrt{\gamma} = \sqrt{\alpha\sigma^2/(2\varepsilon)} \leq 7\times10^{-5}$, so $N^*/r \ll 1$.

When $\gamma \ll 1$, the ceiling correction $\gamma r/N$ is negligible in $J(B,N)$:

$$J(B,N) \approx J_0 + (1-J_0)\cdot\frac{B}{(r+N-1)\tau_0}\cdot\underbrace{\left(1-\frac{\gamma r}{N}\right)}_{\approx 1}$$

This reduces to pure convergence rate, independent of $N$ — **the N-cancellation dominates**, which is exactly what flat pop scaling plots show. The formula is not broken; it correctly predicts that with conservative hyperparameters, $N$ is genuinely indifferent.

### Effect of sigma on the regime

| $\sigma$ | $\gamma = \alpha\sigma^2/(2\varepsilon)$ at $\varepsilon=0.1$ | $\sqrt{\gamma}$ | $N^*/r$ | Regime |
|---|---|---|---|---|
| $10^{-3}$ (ours) | $5\times10^{-9}$ | $7\times10^{-5}$ | $\ll 1$ | Flat pop scaling, $N$ indifferent |
| $10^{-2}$ | $5\times10^{-7}$ | $7\times10^{-4}$ | $\ll 1$ | Still flat |
| $10^{-1}$ | $5\times10^{-5}$ | $7\times10^{-3}$ | small | Weak $N$-dependence |
| $1.0$ | $5\times10^{-3}$ | $0.07$ | $\sim 0.07$ | Genuine $N^*$ emerges |
| O(1/$\sqrt{r}$) = 0.1 | O(1) | O(1) | O(1) | Paper's regime: $N^* \sim r$ |

The formula $N^* = r\sqrt{\gamma}$ becomes quantitatively predictive only when $\sigma \sim O(1/\sqrt{r})$ relative to the problem's natural scale — which is what the paper's normalized coordinate system implicitly assumes.

### The key insight

$$\boxed{N^* = \frac{r\sqrt{\alpha\sigma^2}}{\sqrt{2\varepsilon}}}$$

$N^*$ is controlled by three factors:
1. **$r$** (effective rank) — linear scaling, apples-to-apples
2. **$\sqrt{\alpha\sigma^2}$** — perturbation strength; larger $\sigma$ or $\alpha$ → larger $N^*$
3. **$1/\sqrt{\varepsilon}$** — inverse stability margin; harder modes (closer to boundary) → larger $N^*$

With conservative hyperparameters $\varepsilon \sim O(1)$, $\sqrt{\alpha\sigma^2} \sim 10^{-5}$, so $N^* \approx 0$ — N is indifferent. With aggressive $\sigma \sim 0.1$, $\sqrt{\alpha\sigma^2} \sim 10^{-3}$, $N^* \sim 0.1r = 10$ — a genuine optimum emerges. This connects $\sigma$ calibration directly to whether population size matters at all.

---

## Is N* = r sqrt(alpha sigma squared over 2 epsilon) first-principles or approximate?

Short answer: it is built on approximately seven layers of approximation. Only one intermediate result in the chain is genuinely exact given clearly stated assumptions.

### Layer 1: From the paper (arXiv 2602.00170)

**Approximation 1 — Quadratic expansion near $\theta^*$:**
$$J(\theta^* + x) \approx J(\theta^*) - \frac{1}{2}x^\top \mathbf{H} x$$
Valid only locally near the optimum. Higher-order terms discarded.

**Approximation 2 — Linearized ES dynamics:**
$$x_{i,t+1} = (1-\alpha\lambda_i)x_{i,t} + \frac{\alpha\sigma}{\sqrt{N}}\xi_{i,t}$$
The full ES update is nonlinear. This linearization holds only when already close to $\theta^*$.

**EXACT given 1 and 2 — Stationary variance:**
$$v_{i,\infty} = \frac{\alpha\sigma^2}{N\lambda_i(2-\alpha\lambda_i)}$$
Once you accept the linearized OU process, this is exact — no further approximation.

**EXACT given 1 and 2 — Terminal plateau:**
$$1 - J_\infty(N) = \frac{C}{N}, \quad C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2-\alpha\lambda_i}$$
Also exact given the first two approximations. This is the only genuinely first-principles result in the chain.

### Layer 2: Our synthesis (not in any paper)

**Approximation 3 — Exponential convergence form:**
$$J(T) \approx J_\infty\!\left(1 - e^{-T/\tau}\right) + J_0\,e^{-T/\tau}$$
Exact for linear OU processes. For the actual nonlinear ES dynamics this is a further approximation — convergence may not be single-exponential.

**Approximation 4 — MeZO effective steps substitution:**
$$T \to T_\text{eff} = \frac{B}{r+N-1}$$
Borrows the MeZO blowup factor to account for gradient quality. But MeZO's theorem uses cross-entropy loss and different regularity conditions. Mixing it with the OU-based plateau formula conflates two different frameworks.

**Approximation 5 — Small-budget regime:**
$$e^{-B/(N\tau)} \approx 1 - \frac{B}{N\tau}$$
Valid only when $B \ll N\tau$. At large budgets the formula gives a different $N^*$.

### Layer 3: The N* = r sqrt(alpha sigma squared / 2 epsilon) formula specifically

**Approximation 6 — Uniform eigenvalues:**
$$C = \frac{\alpha\sigma^2}{2}\cdot\frac{r}{\varepsilon}$$
Assumes all $r$ stiff modes have identical eigenvalue $\lambda$ with margin $\varepsilon = 2-\alpha\lambda$. The real spectrum is heterogeneous — near-boundary modes dominate $C$.

**Approximation 7 — $\gamma \ll 1$:**
$$N^* = r(\gamma + \sqrt{\gamma}) \approx r\sqrt{\gamma}$$
Drops the $\gamma$ term relative to $\sqrt{\gamma}$. Valid for our conservative hyperparameters, but is an extra assumption.

### Full approximation stack

| Step | Source | Status |
|------|--------|--------|
| Quadratic expansion near $\theta^*$ | Paper | Approximation (local only) |
| Linearized ES dynamics | Paper | Approximation (near-optimum only) |
| Stationary variance $v_{i,\infty}$ | Paper | **Exact** given above two |
| Terminal plateau $C/N$ | Paper | **Exact** given above two |
| Exponential convergence form | Ours | Approximation |
| MeZO effective steps $T_\text{eff}$ | Ours | Approximation (mixing frameworks) |
| Small-budget expansion | Ours | Approximation |
| Uniform eigenvalue assumption | Ours | Simplification |
| $\gamma \ll 1$ | Ours | Approximation |

### What is actually first-principles

Only one result is rigorous given clearly stated assumptions:

> **Given** quadratic reward near $\theta^*$ and linearized ES dynamics: the terminal plateau is exactly $1 - J_\infty(N) = C/N$ with $C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2-\alpha\lambda_i}$.

Everything after — $J(B,N)$, the $N^*$ quadratic, $r\sqrt{\gamma}$, and $r\sqrt{\alpha\sigma^2/(2\varepsilon)}$ — is structural insight built on approximations stacked on approximations.

**What is robust regardless of the approximations:**
- $N^* = O(r)$ — supported independently by MeZO Theorem 1 and efficiency saturation
- Scaling directions: $N^*$ increases with $\alpha$, $\sigma^2$, $r$, and $1/\varepsilon$
- The γ-regime prediction: small $\gamma$ (conservative hyperparameters) → $N$ indifferent; large $\gamma$ → genuine $N^*$ emerges

**What requires the full approximation stack to hold:**
- The exact prefactor $\sqrt{\gamma}$
- The specific formula $N^* = r\sqrt{\alpha\sigma^2/(2\varepsilon)}$

---

## Are the Near-Optimum Assumptions Valid at the Start of Fine-Tuning?

The theory (quadratic expansion, linearized ES dynamics, stationary variance) formally requires $\theta$ to be near a local maximizer $\theta^*$. At the start of fine-tuning, $\theta_0$ is the pre-trained base model, not $\theta^*$. Does the theory still apply?

### What the theory formally requires

- $J(\theta^* + x) \approx J(\theta^*) - \frac{1}{2}x^\top \mathbf{H} x$ — quadratic landscape near $\theta^*$
- $\|\theta_0 - \theta^*\|$ small enough that the quadratic approximation holds
- Hessian $\mathbf{H}$ approximately constant along the trajectory from $\theta_0$ to $\theta^*$

### The warm-start argument

Pre-trained LLMs provide **basin pre-selection** — they begin inside a wide, well-structured basin of attraction rather than at a random point in $\mathbb{R}^d$. Evidence:

1. **Intrinsic dimensionality ~100**: Li et al. (2018) showed LLMs can be fine-tuned effectively in random subspaces of dimension ~100, meaning the effective optimization landscape is low-dimensional and approximately quadratic along the relevant directions.

2. **LoRA success**: Low-rank adaptation (rank 4–64) works well, implying the fine-tuning trajectory lies in a low-rank subspace — consistent with a quadratic landscape where only a few eigenvectors of $\mathbf{H}$ have large eigenvalues (the stiff modes).

3. **MeZO $n=1$ empirical success**: Zhang et al. (2023) showed MeZO with a single perturbation direction ($n=1$, extreme underestimation) still converges. If the landscape were far from quadratic, this would fail catastrophically.

4. **Short fine-tuning trajectory**: LLM fine-tuning moves $\theta$ by a small fraction of $\|\theta_0\|$. The Hessian at $\theta_0$ and $\theta^*$ are approximately equal, so the quadratic approximation at $\theta_0$ and at $\theta^*$ give similar predictions.

### The two-phase picture

Even if the quadratic assumption fails globally, the fine-tuning trajectory has two distinct phases:

**Phase 1 — Stiff modes (early, transient):**
- Large eigenvalues $\lambda_i$ dominate; the landscape is curved and far from flat
- N-cancellation (Section 3) dominates: progress $\propto B/d$, independent of $N$
- The quadratic assumption is questionable here, but it doesn't matter — $N$ doesn't help anyway
- This phase is short: stiff modes converge in $O(1/\lambda_\text{max})$ steps

**Phase 2 — Flat modes (late, persistent):**
- Small eigenvalues $\lambda_i$ dominate; the landscape is nearly flat near $\theta^*$
- OU theory applies; terminal plateau $C/N$ is the binding constraint
- The quadratic assumption is most valid here
- This is where $N$ matters for the ceiling

The theory is therefore self-consistent: the regime where $N$ matters (Phase 2, flat modes) is precisely the regime where the quadratic approximation is most valid.

### Practical validity table

| Situation | Theory valid? | Why |
|-----------|--------------|-----|
| Base model, early steps | Approximately | Warm-start basin pre-selection; stiff modes dominate anyway |
| Base model, late steps | Yes | Near $\theta^*$ for fine-tuning objective; flat-mode OU applies |
| Random initialization | No | No basin pre-selection; landscape far from quadratic |
| Small model ($\leq$1B) on easy task | Caution | May skip Phase 2 entirely; ceiling effect not binding |
| Large model ($\geq$7B) on hard task | Yes | Long Phase 2; $N^*$ prediction most useful here |

### Open research question

The two-phase picture generates a testable prediction:

> **N should matter more late in training than early.**

Concretely: plot val_acc improvement vs. N separately for early iterations (first 20% of budget) and late iterations (last 20%). If Phase 1 dominates early, the early curves should be flat in $N$. If Phase 2 dominates late, the late curves should show the inverted-U in $N$.

This is a non-trivial prediction that standard pop-scaling plots (aggregated over all iterations) would miss entirely. It would provide the clearest empirical validation of the two-phase theory.

---

## Numerical N_min Derivation for a 1.3B Model

**Setup:** OPT-1.3B ($d = 1.3 \times 10^9$), standard hyperparameters $\sigma = 10^{-3}$, $\alpha = 10^{-4}$, batch size $B = 16$ examples per perturbation evaluation.

### Per-seed degeneracy formula

For binary $\pm 1$ rewards, seed $i$ is **degenerate** when $f(\theta + \sigma\varepsilon_i) = f(\theta - \sigma\varepsilon_i)$, i.e., both perturbed copies get the same reward. Its contribution to the gradient is exactly zero.

**B = 1 (single example per evaluation) — exact:**

$$\delta = p_0^2 + (1-p_0)^2 = 1 - 2p_0(1-p_0)$$

Both correct with prob $p_0^2$; both wrong with prob $(1-p_0)^2$.

**B ≥ 8 (CLT regime) — continuous approximation:**

With $B$ examples, $f^+ = \frac{1}{B}\sum_{b=1}^B r_b^+$ and $f^- = \frac{1}{B}\sum r_b^-$ each have Var $= 4p_0(1-p_0)/B$. Their difference $f^+ - f^-$ has variance $8p_0(1-p_0)/B$ and takes values in steps of $2/B$. Degeneracy = $f^+ = f^-$:

$$\delta_B \approx \frac{1}{2\sqrt{\pi B p_0(1-p_0)}}$$

(PMF at zero of $\text{N}(0, 8Bp_0(1-p_0)/B^2)$ times step $2/B$, simplified.)

### Delta table — per-seed degeneracy probability

| Task scenario | $p_0$ | $B=1$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ |
|---|---|---|---|---|---|---|
| Easy (SST-2 on 1.3B) | 0.70 | 0.580 | 0.218 | 0.154 | 0.109 | 0.077 |
| Medium (BoolQ/RTE) | 0.50 | 0.500 | 0.199 | 0.141 | 0.100 | 0.071 |
| Hard (TREC 6-class) | 0.30 | 0.580 | 0.218 | 0.154 | 0.109 | 0.077 |

Note: $p_0 = 0.3$ and $p_0 = 0.7$ give the same $\delta$ (the formula is symmetric around 0.5) — degeneracy is hardest near $p_0 = 0.5$.

### N_min table

$N_\text{min}$ = smallest $N$ with at least one informative seed per step with probability $> \tau$:

$$N_\text{min}(\tau) = \left\lceil \frac{\log(1-\tau)}{\log \delta} \right\rceil$$

**$\tau = 0.95$ (95% coverage):**

| Task | $B=1$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ |
|---|---|---|---|---|---|
| Easy ($p_0=0.70$) | 6 | 2 | 2 | 2 | 2 |
| Medium ($p_0=0.50$) | 5 | 2 | 2 | 2 | 2 |
| Hard ($p_0=0.30$) | 6 | 2 | 2 | 2 | 2 |

**$\tau = 0.99$ (99% coverage):**

| Task | $B=1$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ |
|---|---|---|---|---|---|
| Easy ($p_0=0.70$) | 9 | 4 | 3 | 3 | 2 |
| Medium ($p_0=0.50$) | 7 | 3 | 3 | 2 | 2 |
| Hard ($p_0=0.30$) | 9 | 4 | 3 | 3 | 2 |

**Bottom line**: $N_\text{min} \in [2, 9]$ across all realistic scenarios. Even in the extreme worst case ($B=1$, hardest binary task), $N_\text{min} \leq 9$.

### What N~30 means — correcting the sandwich interpretation

The hypothesis that "N~30 works best empirically → $N_\text{min} \approx 16$–30" conflates the two sandwich boundaries:

| Boundary | Value for 1.3B | What it means |
|---|---|---|
| $N_\text{min}$ (degeneracy lower bound) | **2–7** | Below this: gradient too noisy to be useful |
| $N^*$ (from $\gamma$ formula) | **≈ 0** (indifferent) | Ceiling theory: optimal $N$ in plateau regime |
| $r$ (effective rank upper bound) | **~100** | Above this: N-cancellation kills marginal gain |
| Empirical "works well" | **N~30** | In the middle of the flat zone |

With $\alpha = 10^{-4}$, $\sigma = 10^{-3}$, $\varepsilon \approx 1$ (safe stability margin):

$$\gamma = \frac{\alpha\sigma^2}{2\varepsilon} = \frac{10^{-4} \cdot 10^{-6}}{2} = 5 \times 10^{-11}$$

$$N^* = r\sqrt{\gamma} = 100 \cdot \sqrt{5 \times 10^{-11}} \approx 0.0007$$

This confirms $\gamma \ll 1$ — the ceiling correction is negligible and $N$ is genuinely indifferent across $[N_\text{min}, r] = [2\text{–}7,\ 100]$.

### Effective seeds with N=30, B=16

Even though $\delta \approx 0.15$ per seed, at $N=30$ about 85% of seeds are informative:

| Task | $\delta$ (B=16) | $N_\text{eff} = N(1-\delta)$ at $N=30$ |
|---|---|---|
| Easy ($p_0=0.70$) | 0.154 | 25.4 |
| Medium ($p_0=0.50$) | 0.141 | 25.8 |
| Hard ($p_0=0.30$) | 0.154 | 25.4 |

Degeneracy wastes ~4–5 seeds out of 30 — a 15% penalty, not a convergence barrier.

### Key insight

> **$N_\text{min}$ is 2–7 for a 1.3B model with standard hyperparameters, not 16–30.**
>
> The empirical observation that N~30 "works well" is **upper-bound evidence** — it shows where $r$ (the effective rank ceiling) is, not where $N_\text{min}$ is. N=30 sits in the middle of the flat zone $[2\text{–}7, 100]$.
>
> Degeneracy (binary reward) costs roughly $\delta \approx 15\%$ of seeds per step, but this is a constant-factor penalty absorbed by the $N$-cancellation budget, not an asymptotic barrier. The only scenario where $N_\text{min}$ would reach 16–30 is single-example evaluation ($B=1$) with a 99.9%+ coverage requirement — far stricter than any practical training setup.

---

## Hyperparameter Coupling When N Changes

### What was held fixed in all prior derivations

Every result so far — N-cancellation, C/N ceiling, N* from γ, N_min from degeneracy — assumed:
- $\sigma$ fixed across N comparisons
- $\alpha$ fixed across N comparisons
- Total budget $B = N \times T$ fixed

This section asks: when you increase N, should you also adjust $\sigma$ or $\alpha$? And what happens to performance if you don't?

### What sigma controls (and why it doesn't need to scale with N)

The antithetic ES gradient estimate for one seed:

$$\hat{g}_i = \frac{f(\theta + \sigma\varepsilon_i) - f(\theta - \sigma\varepsilon_i)}{2\sigma}\,\varepsilon_i$$

Taylor-expand the numerator: $f(\theta + \sigma\varepsilon) - f(\theta - \sigma\varepsilon) = 2\sigma(\varepsilon^\top \nabla J) + O(\sigma^3)$.

Dividing by $2\sigma$: $\hat{g}_i = (\varepsilon_i^\top \nabla J)\varepsilon_i + O(\sigma^2)$.

**The $\sigma$ cancels exactly to leading order.** The variance of $\hat{g}_i$ is therefore:

$$\mathrm{Var}(\hat{g}_i) \approx \|\nabla J\|^2 d \quad \text{(independent of }\sigma\text{)}$$

What $\sigma$ actually controls:

| Effect | Formula | Notes |
|---|---|---|
| Estimator variance | $\approx \|\nabla J\|^2 d$ | **Independent of σ** |
| Estimator bias | $\nabla J_\sigma - \nabla J \approx \frac{\sigma^2}{2}\nabla(\Delta J)$ | Grows as σ² |
| Target landscape | Optimizing $J_\sigma$ (blurred $J$), not $J$ | Larger σ = smoother, less faithful |
| Degeneracy probability | $\delta_B \approx 1/(2\sqrt{\pi B p_0(1-p_0)})$ | **Independent of σ** |
| OU terminal plateau | $C = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2-\alpha\lambda_i}$ | Scales as σ² |

**Conclusion on σ**: $\sigma$ does not need to scale with $N$. Changing $\sigma$ adjusts which landscape you optimize and the asymptotic plateau, not the gradient estimator variance.

### What alpha controls (and why it must scale with N)

From the MeZO convergence bound for $L$-smooth $f$, at fixed total budget $B = NT$:

$$\min_t \|\nabla J(\theta_t)\|^2 \leq \underbrace{\frac{\Delta f \cdot N}{\alpha B}}_{\text{step-count cost: grows with }N} + \underbrace{\frac{\alpha \sigma^2 d}{N}}_{\text{variance floor: shrinks with }N}$$

The step-count term grows with N (larger N → fewer steps → less progress per budget). The variance floor shrinks with N (larger N → better gradient estimate per step). These exactly trade off when $\alpha$ is scaled linearly with $N$.

#### Numerical illustration (1.3B model, $\alpha_0 = 10^{-4}$, $\sigma = 10^{-3}$, $B = 1000$)

| N | Fixed α = 10⁻⁴ (suboptimal) | Scaled α = N × 10⁻⁴ (optimal) |
|---|---|---|
| 1 | bound = **5.13** (best) | bound = 5.13 |
| 4 | bound = 20.03 (4× worse) | bound = 5.13 |
| 16 | bound = 80.01 (16× worse) | bound = 5.13 |
| 32 | bound = 160.004 (32× worse) | bound = 5.13 |
| 64 | bound = 320.002 (64× worse) | bound = 5.13 |

With **fixed α**: N=1 is always best. The variance floor (≈ 0.13) is negligible compared to the step-count penalty (≈ 5×N), so you gain almost nothing from variance reduction while losing badly from having fewer steps.

With **scaled α ∝ N**: both terms are identical across all N — perfect N-cancellation. This is the correct regime for comparing N values fairly.

### What happens in practice (why empirical pop-scaling shows an inverted-U)

Standard ES experiments sweep N with **fixed** $\sigma$, $\alpha$, and total budget $B$. Under fixed α:

- **Small N** (1–4): many steps but high variance. $\alpha$ is tuned for this regime.
- **Medium N** (~30): α happens to be reasonably tuned; the plateau benefit $1 - C/N$ partially kicks in; step-count penalty not yet extreme.
- **Large N** (>64): few steps with same $\alpha$ → severe step-count penalty → worse.

The "inverted-U" with optimum at N~30 is therefore partially an artifact of the fixed α regime, not a fundamental property of population size. The true test is:

> **Sweep N while also scaling α ∝ N. If pop-scaling curve is still non-flat, that's genuine. If it flattens, the prior "optimal N" was an α-tuning artifact.**

### Summary of coupling rules

| Hyperparameter | Scale with N? | Why | If you don't scale |
|---|---|---|---|
| $\alpha$ (learning rate) | **Yes — linearly** ($\alpha_N = N\alpha_0$) | Restores N-cancellation; maintains step quality | Larger N becomes progressively suboptimal |
| $\sigma$ (perturbation) | **No** | σ cancels in estimator; controls bias not variance | No effect on variance; changing σ changes which landscape you optimize |
| $B$ (examples per eval) | No | Controls degeneracy and estimator accuracy, not N-scaling | Separate tuning axis |

### Effect on the terminal plateau when α scales with N

If $\alpha_N = N\alpha_0$, the plateau constant becomes:

$$C(N) = \frac{\alpha_N \sigma^2}{2}\sum_i \frac{1}{2 - \alpha_N\lambda_i} = \frac{N\alpha_0 \sigma^2}{2}\sum_i \frac{1}{2 - N\alpha_0\lambda_i}$$

The plateau is $1 - C(N)/N = 1 - \frac{\alpha_0\sigma^2}{2}\sum_i \frac{1}{2 - N\alpha_0\lambda_i}$.

For small $N\alpha_0\lambda_i$ (all modes far from instability), this $\approx 1 - \frac{\alpha_0\sigma^2}{2}\sum_i \frac{1}{2} = $ constant — independent of N. The ceiling doesn't change.

But as N grows and $N\alpha_0\lambda_i \to 2$ (stability boundary), the ceiling degrades — you hit instability before benefiting from the larger population. This sets a hard upper limit on N under scaled-α rule: $N < 2/(\alpha_0 \lambda_\text{max})$.

> **Key insight**: σ is a free axis — adjust it to control bias/landscape without affecting variance. α must co-scale with N or you pay a linear penalty in convergence speed. The "optimal N~30" seen empirically is as much about where α is well-calibrated as it is about the fundamental ES population dynamics.

---

## Antithetic Sampling vs One-Sided Sampling: Full Theoretical Comparison

### Formal definitions

**One-sided estimator** (N seeds, N+1 forward passes per step):
$$\hat{g}^{\text{one}}_i = \frac{f(\theta + \sigma\varepsilon_i) - f(\theta)}{\sigma}\,\varepsilon_i$$

**Antithetic estimator** (N seed pairs, 2N forward passes per step):
$$\hat{g}^{\text{anti}}_i = \frac{f(\theta + \sigma\varepsilon_i) - f(\theta - \sigma\varepsilon_i)}{2\sigma}\,\varepsilon_i$$

### Bias — identical for both

Apply Stein's lemma to each:
$$E[\hat{g}^{\text{one}}] = \frac{1}{\sigma}E[f(\theta+\sigma\varepsilon)\varepsilon] = \nabla J_\sigma(\theta)$$
$$E[\hat{g}^{\text{anti}}] = \frac{1}{2\sigma}\bigl[E[f(\theta+\sigma\varepsilon)\varepsilon] - E[f(\theta-\sigma\varepsilon)\varepsilon]\bigr] = \frac{1}{2\sigma}[\sigma\nabla J_\sigma + \sigma\nabla J_\sigma] = \nabla J_\sigma(\theta)$$

Both are unbiased for the **smoothed gradient** $\nabla J_\sigma$, not the true gradient $\nabla J$. The bias relative to $\nabla J$ is:
$$E[\hat{g}] - \nabla J = \frac{\sigma^2}{2}\nabla(\Delta J) + O(\sigma^4) \quad \text{(same for both, to all orders)}$$

### Variance — the Taylor expansion explains everything

Expand the numerators around $\theta$:

$$f(\theta + \sigma\varepsilon) = f(\theta) + \sigma(\varepsilon^\top\nabla J) + \frac{\sigma^2}{2}(\varepsilon^\top H\varepsilon) + \frac{\sigma^3}{6}T_3[\varepsilon,\varepsilon,\varepsilon] + \cdots$$
$$f(\theta - \sigma\varepsilon) = f(\theta) - \sigma(\varepsilon^\top\nabla J) + \frac{\sigma^2}{2}(\varepsilon^\top H\varepsilon) - \frac{\sigma^3}{6}T_3[\varepsilon,\varepsilon,\varepsilon] + \cdots$$

**Antithetic numerator** $f^+ - f^-$: ALL even-order terms cancel ($\sigma^0, \sigma^2, \sigma^4, \ldots$):
$$f^+ - f^- = 2\sigma(\varepsilon^\top\nabla J) + \frac{\sigma^3}{3}T_3[\varepsilon,\varepsilon,\varepsilon] + O(\sigma^5)$$
$$\hat{g}^{\text{anti}}_i = (\varepsilon_i^\top\nabla J)\varepsilon_i + O(\sigma^2) \quad \leftarrow \text{Hessian term gone}$$

**One-sided numerator** $f^+ - f(\theta)$: even-order terms survive:
$$f^+ - f(\theta) = \sigma(\varepsilon^\top\nabla J) + \frac{\sigma^2}{2}(\varepsilon^\top H\varepsilon) + O(\sigma^3)$$
$$\hat{g}^{\text{one}}_i = (\varepsilon_i^\top\nabla J)\varepsilon_i + \underbrace{\frac{\sigma}{2}(\varepsilon_i^\top H\varepsilon_i)\varepsilon_i}_{\text{zero mean, nonzero variance}} + O(\sigma^2)$$

**Per-seed variance:**

For antithetic, using $E[(\varepsilon^\top a)^2 \varepsilon\varepsilon^\top] = \|a\|^2 I + 2aa^\top$:
$$\text{Var}(\hat{g}^{\text{anti}}_i) \approx \|\nabla J\|^2 d$$

For one-sided, the Hessian term adds:
$$\text{Var}(\hat{g}^{\text{one}}_i) \approx \|\nabla J\|^2 d + \underbrace{\frac{\sigma^2}{2}\|\mathbf{H}\|_F^2 \cdot d}_{\text{Hessian noise}}$$

### Which estimator wins at equal compute?

At total budget of $2N$ forward passes:
- Antithetic($N$ pairs): Var $\approx \|\nabla J\|^2 d / N$
- One-sided($2N$ seeds): Var $\approx (\|\nabla J\|^2 d + \frac{\sigma^2}{2}\|\mathbf{H}\|_F^2 d) / (2N)$

One-sided wins when $\|\nabla J\|^2 > \frac{\sigma^2}{2}\|\mathbf{H}\|_F^2$, i.e., when $\|\mathbf{H}\|_F < \|\nabla J\|/\sigma \cdot \sqrt{2}$.

**Critical correction — the Hessian for LLMs is sparse:**

$$\|\mathbf{H}\|_F^2 = \sum_i \lambda_i^2 = r_\text{eff} \cdot \bar{\lambda}^2 \quad \text{(not } d\cdot\bar{\lambda}^2 \text{)}$$

Pre-trained LLMs have $r_\text{eff} \approx 100$ non-negligible eigenvalues:

| Scenario | $\|\mathbf{H}\|_F^2$ | $\Delta\text{Var}/\text{Var}$ | Winner at equal compute |
|---|---|---|---|
| Very flat ($\bar\lambda = 10^{-3}$) | $10^{-4}$ | $10^{-9}$ | One-sided (by ~50%) |
| Moderate ($\bar\lambda = 10^{-2}$) | $10^{-2}$ | $10^{-7}$ | One-sided (by ~50%) |
| Strong ($\bar\lambda = 10^{-1}$) | $1$ | $10^{-5}$ | One-sided (by ~50%) |
| Dense (wrong: $d\bar\lambda^2$) | $1.3\times10^5$ | $1.6$ | Antithetic (by 31%) |

**Crossover eigenvalue** (where methods are equal): $\bar\lambda = \sqrt{2\|\nabla J\|^2/\sigma^2 / r_\text{eff}} \approx 0.028$ for $\sigma=10^{-3}$, $p_0=0.6$.

Since LLM stiff-mode eigenvalues are typically $\bar\lambda \sim 10^{-3}$ to $10^{-2}$, **one-sided wins at equal compute for all realistic LLM fine-tuning settings**.

### Connection to all prior theory

**N-cancellation:**
Both estimators satisfy N-cancellation with $\alpha \propto N$. At equal budget ($2N$ FP), one-sided has effective seed count $2N$, so its α-scaling is $\alpha_\text{one} = 2N\alpha_0$ vs $\alpha_\text{anti} = N\alpha_0$. Under correct scaling, both give identical total progress per budget. ✓

**Terminal plateau $C/N$:**
$$C^\text{anti} = \frac{\alpha\sigma^2}{2}\sum_i \frac{1}{2-\alpha\lambda_i}$$
$$C^\text{one} = C^\text{anti} + \frac{\alpha\sigma^4}{4}\sum_i \frac{\|\mathbf{H}_i\|^2}{\lambda_i(2-\alpha\lambda_i)} \approx C^\text{anti} \quad (\sigma^4 \text{ correction is negligible})$$

With $\sigma=10^{-3}$: $C^\text{one} \approx C^\text{anti} \times (1 + O(10^{-2}))$. Plateau is virtually identical. ✓

**Degeneracy:**
Both estimators have the same per-seed degeneracy probability:
$$\delta = p_0^2 + (1-p_0)^2 \quad (B=1), \qquad \delta \approx \frac{1}{2\sqrt{\pi B p_0(1-p_0)}} \quad (B\geq 8)$$

Subtle caveat: if the one-sided baseline $f(\theta)$ is re-evaluated stochastically each step (not fixed), the effective degeneracy is identical. If $f(\theta)$ is a stale fixed reference, it may introduce a constant offset that slightly increases effective degeneracy when the current θ has drifted from the reference. In practice this is minor. ✓

**σ-dependence:**
| | Antithetic | One-sided |
|---|---|---|
| Variance vs σ | Independent (σ cancels in $f^+ - f^-$) | Grows as $\sigma^2\|\mathbf{H}\|_F^2$ |
| Bias | Grows as $\sigma^2$ | Grows as $\sigma^2$ (same) |
| Degeneracy | Independent | Independent |

One-sided has a genuine $\sigma$-dependent variance source that antithetic eliminates. This is the one qualitative advantage of antithetic.

**Reward type interaction (binary vs CE):**
| | Binary reward | CE reward |
|---|---|---|
| Landscape roughness | High (jagged decision boundaries) | Low (smooth probability output) |
| $\|\mathbf{H}\|_F^2$ | Larger | Smaller |
| Antithetic advantage | More important | Less important |
| Crossover $\bar\lambda$ | Easier to exceed | Harder to exceed |

Binary reward makes the landscape more jagged → larger $\|\mathbf{H}\|_F^2$ → antithetic's advantage over one-sided grows.

### Summary table

| Property | One-sided | Antithetic |
|---|---|---|
| Unbiased for | $\nabla J_\sigma$ | $\nabla J_\sigma$ |
| Bias vs $\nabla J$ | $\frac{\sigma^2}{2}\nabla\Delta J + O(\sigma^4)$ | **identical** |
| Even-order Taylor noise | Retained ($\sigma^2$ Hessian term) | Cancelled |
| Per-seed variance | $\|\nabla J\|^2 d + \sigma^2\|\mathbf{H}\|_F^2 d/2$ | $\|\nabla J\|^2 d$ |
| FP cost for $N$ seeds | $N+1 \approx N$ | $2N$ |
| Variance at equal FP budget | $\approx$ half of antithetic (for LLMs) | baseline |
| Winner at equal compute (LLM, $\sigma\leq10^{-3}$) | **One-sided** | — |
| Winner at equal compute (large $\sigma$, curved) | — | **Antithetic** |
| N-cancellation | ✓ | ✓ |
| Terminal plateau | $\approx$ identical | baseline |
| Degeneracy | $\approx$ identical | baseline |
| σ-dependence of variance | Yes ($\sigma^2\|\mathbf{H}\|_F^2$) | No (to leading order) |

> **Key insight**: Antithetic sampling's value is in eliminating the even-order Hessian noise $\frac{\sigma}{2}(\varepsilon^\top H\varepsilon)\varepsilon$ from the gradient estimate. For LLM fine-tuning, this correction is negligible ($\ll 0.01\%$) because the LLM Hessian is sparse ($r_\text{eff}\approx100$, not $d=10^9$). One-sided sampling with doubled seed count ($2N$) achieves ~50% lower variance at the same compute cost.
>
> Antithetic is the right choice when $\sigma$ is large (classic ES, $\sigma\sim0.1$) or when the landscape is genuinely curved (large $\|\mathbf{H}\|_F$). For LLM fine-tuning with $\sigma\sim10^{-3}$, one-sided sampling is computationally dominant — a finding that suggests current practice (using antithetic by default) may be leaving a ~2× compute efficiency gain on the table.

---

## Rademacher vs Gaussian Noise: Full Theoretical Comparison

### Definitions and moment structure

| | Gaussian $\varepsilon \sim \mathcal{N}(0, I_d)$ | Rademacher $\varepsilon_i \in \{-1,+1\}$ |
|---|---|---|
| $E[\varepsilon_i]$ | 0 | 0 |
| $E[\varepsilon_i^2]$ | 1 | 1 |
| $E[\varepsilon_i\varepsilon_j]$ | $\delta_{ij}$ | $\delta_{ij}$ |
| $E[\varepsilon_i^4]$ | 3 | **1** |
| Kurtosis $\kappa_4$ | 0 | **−2** (sub-Gaussian) |
| $\|\sigma\varepsilon\|^2$ | $\sigma^2\chi^2(d)$, mean $\sigma^2 d$, std $\sigma^2\sqrt{2d}$ | **$\sigma^2 d$ exactly** |

Both distributions share identical first and second moments. They differ starting at the fourth moment through kurtosis.

### Bias — identical to O(σ²)

The gradient estimator bias analysis uses $E[\varepsilon_i\varepsilon_j] = \delta_{ij}$ — both distributions satisfy this identically. So both estimators (antithetic or one-sided) are unbiased for $\nabla J_\sigma$ and share the same leading bias:

$$E[\hat{g}] - \nabla J = \frac{\sigma^2}{2}\nabla(\Delta J) + O(\sigma^4 \cdot \kappa_4)$$

The $O(\sigma^4)$ correction differs through kurtosis: Rademacher has $\kappa_4 = -2$ vs Gaussian $\kappa_4 = 0$. For $\sigma = 10^{-3}$: $\sigma^4 = 10^{-12}$ — completely negligible.

**Bias is identical for both distributions for all practical purposes.**

### Variance — the fourth-moment calculation

For the antithetic estimator leading term $\hat{g}_l = (\varepsilon^\top\nabla J)\varepsilon_l$:

$$E[(\hat{g}_l)^2] = \sum_{ij}(\nabla J)_i(\nabla J)_j\, E[\varepsilon_i\varepsilon_j\varepsilon_l^2]$$

**Gaussian** (Wick's theorem): $E[\varepsilon_i\varepsilon_j\varepsilon_l^2] = \delta_{ij} + 2\delta_{il}\delta_{jl}$
$$\text{Var}_\text{Gauss}(\hat{g}_l) = \|\nabla J\|^2 + (\nabla J)_l^2$$

**Rademacher** ($\varepsilon_l^2 = 1$ always, so $E[\varepsilon_i\varepsilon_j\varepsilon_l^2] = E[\varepsilon_i\varepsilon_j] = \delta_{ij}$):
$$\text{Var}_\text{Rad}(\hat{g}_l) = \|\nabla J\|^2 - (\nabla J)_l^2$$

Summed over all $d$ components (total trace variance per seed):

$$\text{Tr}[\text{Var}_\text{Gauss}] = (d+1)\|\nabla J\|^2, \qquad \text{Tr}[\text{Var}_\text{Rad}] = (d-1)\|\nabla J\|^2$$

**Difference: exactly $2\|\nabla J\|^2$ — a constant, independent of $d$.**

| Model | Relative variance difference $2/d$ |
|---|---|
| OPT-350M | $5.7 \times 10^{-9}$ |
| OPT-1.3B | $1.5 \times 10^{-9}$ |
| Llama-7B | $2.9 \times 10^{-10}$ |
| Llama-70B | $2.9 \times 10^{-11}$ |

**For any LLM with $d \geq 350\text{M}$: the variance advantage of Rademacher is less than $10^{-8}$ relative — completely imperceptible.**

### Blowup factor and N-cancellation

MeZO blowup for Gaussian (n=N seeds): $(d+N-1)/N$

Rademacher antithetic (N seed pairs): $(d-1+N)/N$

Difference: $2/N$ — negligible for large $d$. The MeZO learning rate scaling rule $\eta_\text{ZO} \approx N/d \cdot \eta_\text{SGD}$ holds equally. **N-cancellation is identical for both distributions.**

### One-sided Hessian noise — Rademacher's only structural advantage

For the one-sided estimator's Hessian noise term $\frac{\sigma}{2}(\varepsilon^\top H\varepsilon)\varepsilon_l$:

$$E_\text{Gauss}[(\varepsilon^\top H\varepsilon)^2] = 2\|\mathbf{H}\|_F^2 + (\text{Tr}\,H)^2$$

$$E_\text{Rad}[(\varepsilon^\top H\varepsilon)^2] = 2\sum_{i\neq j}H_{ij}^2 + (\text{Tr}\,H)^2 = E_\text{Gauss}[\cdots] - 2\sum_i H_{ii}^2$$

**Rademacher eliminates the diagonal Hessian term $2\sum_i H_{ii}^2$ exactly.**

For diagonal $H$ (all off-diagonal entries zero):
- Gaussian: $2\|H\|_F^2 + (\text{Tr}\,H)^2$
- Rademacher: only $(\text{Tr}\,H)^2$ — the entire $\|H\|_F^2$ term is gone

For sparse Hessian ($r_\text{eff}$ non-zero modes, eigenvalue $\bar\lambda$):

$$\text{Ratio: } \frac{(\text{Tr}\,H)^2}{2\|H\|_F^2 + (\text{Tr}\,H)^2} = \frac{r_\text{eff}}{r_\text{eff}+2}$$

| $r_\text{eff}$ | Rademacher Hessian reduction |
|---|---|
| 1 | 33% (significant!) |
| 10 | 83% |
| 100 | **98%** (negligible advantage) |
| 1000 | 99.8% |

For LLMs with $r_\text{eff} \approx 100$: Rademacher reduces one-sided Hessian noise by 98% relative to Gaussian's 100% — **only a 2% advantage, negligible.**

### Perturbation norm stability

$$\text{CV}(\|\sigma\varepsilon\|) = \sqrt{2/d}$$

| Model | Gaussian norm CV | Rademacher norm CV |
|---|---|---|
| OPT-350M | $7.6\times10^{-5}$ | 0 |
| OPT-1.3B | $3.9\times10^{-5}$ | 0 |
| Llama-7B | $1.7\times10^{-5}$ | 0 |

Rademacher has fixed perturbation norm $\sigma\sqrt{d}$ exactly. But Gaussian's norm is already extremely concentrated for large $d$ — the difference is negligible.

### Degeneracy

Both: $\delta \approx p_0^2 + (1-p_0)^2$ ($B=1$), $\approx 1/(2\sqrt{\pi B p_0(1-p_0)})$ ($B\geq 8$).

Subtle distinction: Rademacher $\varepsilon \in \{-1,+1\}^d$ explores only $2^d$ discrete hypercube corners. For $d = 350\text{M}$: $2^{350\text{M}}$ directions — far denser than any sampling could reach. **Effectively identical direction coverage for any LLM.**

### When Rademacher should fail vs Gaussian (theoretical predictions)

| Setting | Prediction | Reason |
|---|---|---|
| Any LLM ($d \geq 350\text{M}$), antithetic | **Indifferent** | Variance diff $2/d \approx 0$ |
| Any LLM, one-sided, diagonal H | **Indifferent** | Hessian advantage only $2/(r_\text{eff}+2) \approx 2\%$ |
| $d \leq 10$ (classic control tasks) | Rademacher might fail | Only $2^d$ discrete directions, insufficient coverage |
| Large $\sigma \geq 0.1$ | Rademacher might fail | Bounded $|\varepsilon_i|=1$ prevents large per-component steps; Gaussian can escape via heavy tail |
| Dense off-diagonal Hessian | Rademacher loses advantage | Only diagonal $H_{ii}^2$ terms eliminated; off-diagonal cross-terms remain |
| Hyperparameter tuning sensitivity | Rademacher slightly easier | Fixed norm → $\sigma$ is easier to tune |

### Why OPT-350M on RTE "worked kind of well" with Rademacher

Theory says both methods should be **indifferent** for any $d \geq 350\text{M}$: variance difference $5.7 \times 10^{-9}$, norm CV $7.6 \times 10^{-5}$, degeneracy identical. None of these can explain an empirical advantage.

Plausible explanations (in decreasing plausibility):
1. **Statistical noise**: the experiment was within the measurement noise floor — no real difference
2. **Hyperparameter sensitivity**: Rademacher's fixed norm makes $\sigma$ slightly easier to tune, so whatever $\sigma$ was chosen happened to work better
3. **Sub-Gaussian tails** ($\kappa_4 = -2$): no extreme per-parameter perturbations → slightly more stable early updates
4. **RTE-specific structure**: 2-class NLI has simple decision boundary → approximately diagonal loss Hessian → Rademacher's 2% diagonal-H advantage just barely helps

### Connection to all prior theory

| Prior result | Gaussian version | Rademacher version |
|---|---|---|
| N-cancellation | $\alpha \propto N$ | **identical** |
| Terminal plateau $C/N$ | $C = \frac{\alpha\sigma^2}{2}\sum\frac{1}{2-\alpha\lambda_i}$ | **identical** (uses $E[\varepsilon\varepsilon^\top]=I$ only) |
| Degeneracy $N_\text{min}$ | 2–7 | **identical** |
| $\alpha$ coupling with $N$ | linear scaling rule | **identical** |
| Antithetic vs one-sided | one-sided better at equal compute | **identical conclusion** |
| Binary vs CE reward | CE reduces $N_\text{min}$, smooths landscape | **identical for both noise types** |

> **Key insight**: Gaussian and Rademacher noise are theoretically indistinguishable for LLM fine-tuning ($d \geq 350\text{M}$) — the variance difference is $2/d \approx 10^{-9}$ relative, below any meaningful measurement threshold. Every prior result in this document (N-cancellation, $C/N$ plateau, $N_\text{min}$, $\alpha$-scaling, antithetic vs one-sided) holds identically for both distributions.
>
> The only regime where the choice matters is $d \leq 10$ (classical ES) or very large $\sigma \geq 0.1$ (where Gaussian's heavier tails per dimension matter for exploration). For all LLM fine-tuning experiments: **pick whichever is cheaper to sample** — they are the same.

---

## N-Cancellation During Active Descent: Does 2N Beat N?

### The cancellation is SNR-independent

A natural question: the N-cancellation was derived for the plateau regime. Does it still hold during **active descent** — when the loss is dropping, the gradient is large, and the model is making real progress? Or does 2N provide a genuine benefit there?

The cancellation holds throughout, and is actually *stronger* during active descent.

The entire cancellation is pure algebra:

$$\text{Progress}(N) = T \cdot \frac{N}{d}\left(\|\nabla f\|^2 - 1\right) = \frac{B}{N} \cdot \frac{N}{d}\left(\|\nabla f\|^2 - 1\right) = \frac{B}{d}\left(\|\nabla f\|^2 - 1\right)$$

The gradient norm $\|\nabla f\|^2$ appears symmetrically for $N$ and $2N$ — it does not interact with the cancellation step. The derivation does not require SNR $\gg 1$.

### The second-order correction shrinks with large gradient

The residual marginal gain from $N \to 2N$ comes from the second-order bias term in $\mathbb{E}[\|\hat{g}\|^2]$:

$$\frac{\text{Progress}(2N) - \text{Progress}(N)}{\text{Progress}(N)} \lesssim \frac{r}{2N^2 \cdot \text{SNR}}$$

where $\text{SNR} \propto \|\nabla f\|^2$. During active descent $\|\nabla f\|$ is large, so SNR is large, and the marginal gain is even smaller than in the plateau.

**Numerical example** ($r = 100$, $N = 8$, $\|\nabla f\| = 300$, $B = 16$, $\sigma = 10^{-3}$, $p_0 = 0.5$):

$$\text{SNR} = \frac{16 \times 10^{-6} \times 9 \times 10^4}{0.5} \approx 2.88$$

$$\text{Marginal gain} \lesssim \frac{100}{2 \times 64 \times 2.88} \approx 0.27\%$$

Compared to the plateau (SNR $\approx 0.3$, marginal gain $\approx 3\%$), the active descent regime is 10× less sensitive to $N$.

### The one exception: crossing $N_\text{min}$

Below $N_\text{min}$, the progress term $(\|\nabla f\|^2 - 1)$ goes **negative** — noise dominates and each update increases the loss in expectation. Going from $N$ to $2N$ can flip the sign of expected progress from negative to positive. This is the only phase transition where doubling $N$ has a qualitative effect, not just a marginal one.

| Regime | Does $2N$ beat $N$? | Magnitude |
|---|---|---|
| $N < N_\text{min}$ | Yes — qualitative flip | Sign change: negative → positive progress |
| $N_\text{min} \leq N$, active descent | No | $\leq 0.27\%$ gain (high SNR) |
| $N_\text{min} \leq N$, near convergence | No | $\leq 3\%$ gain (low SNR) |
| $N > r$ | $2N$ worse | Steps collapse, over-sampling |

**Practical implication**: during active descent, budget is strictly better spent on more steps (smaller $N$) than more seeds. The only investment in larger $N$ worth making is the single jump from below $N_\text{min}$ to above it.

---

## What Breaks When $\eta$ Does Not Scale With $N$

The N-cancellation assumes $\eta^*(N) = N/d$. In practice we use a fixed $\eta$. This breaks the cancellation:

At fixed $\eta$, total progress over $T = B/N$ steps:

$$\text{Progress}(N) = \frac{B}{N}\left[\eta\|\nabla f\|^2 - \frac{2\eta^2 Ld\|\nabla f\|^2}{N}\right] = B\|\nabla f\|^2\left[\frac{\eta}{N} - \frac{2\eta^2 Ld}{N^2}\right]$$

$N$ does not cancel. The ratio of progress for $2N$ vs $N$:

$$\frac{\text{Progress}(2N)}{\text{Progress}(N)} = \frac{1 - u}{2(1 - 2u)} \approx \frac{1}{2}(1+u) < 1 \quad \text{where } u = \frac{\eta Ld}{N}$$

**Smaller $N$ wins** at fixed $\eta$. More steps with the same per-step learning rate outperform fewer steps with a larger population, because each step contributes the same $\eta\|\nabla f\|^2$ and you simply get more of them.

This means the theoretical plateau — flat across $[N_\text{min}, r]$ — becomes a **monotone slope** favoring smaller $N$ under fixed $\eta$. The slope is shallow (proportional to $u = \eta Ld/N$), so it falls within measurement noise in most experiments, which is why pop-scaling curves look flat even under fixed $\eta$. But the true optimal under fixed $\eta$ is always $N = N_\text{min}$, not an interior point.

### Convergence rate: $N$ vs $2N$ at fixed $\eta$

Both $N$ and $2N$ converge to the **same $\theta^*$** — the fixed $\eta$ does not change the destination, only the speed. The asymptotic plateau is $C/N$ for population $N$ and $C/(2N)$ for population $2N$, so $2N$ actually has a slightly lower floor. But it gets there more slowly.

The convergence rate for each is approximately exponential in steps:

$$\text{Loss gap}(t) \approx \Delta_0 \cdot e^{-t / \tau}$$

where $\tau \propto 1/(\eta\|\nabla f\|^2)$ is the same for both $N$ and $2N$ at fixed $\eta$ (same step size, same per-step progress). The difference is that $N$ takes $T = B/N$ steps per budget and $2N$ takes $T/2 = B/(2N)$ steps. Substituting:

$$\text{Loss gap after budget } B: \quad \begin{cases} N: & \Delta_0 \cdot e^{-B/(N\tau)} \\ 2N: & \Delta_0 \cdot e^{-B/(2N\tau)} \end{cases}$$

Since $e^{-B/(N\tau)} < e^{-B/(2N\tau)}$ (larger exponent), **$N$ closes the loss gap faster at every budget checkpoint**.

The crossover in which is better depends on whether you care about speed or asymptotic floor:

| What you measure | Winner | Why |
|---|---|---|
| Loss after fixed budget $B$ (small–medium) | $N$ | More steps, faster exponential decay |
| Asymptotic plateau (infinite budget) | $2N$ | Lower floor $C/(2N) < C/N$ |
| Loss after fixed budget $B$ (very large, both converged) | $2N$ | Has reached its lower floor; $N$ stuck at higher $C/N$ |

**The crossover budget** — where $2N$ starts to win — is when $N$ has saturated its plateau but $2N$ has not yet reached its lower one:

$$B_\text{crossover} \approx N\tau \cdot \log\frac{\Delta_0}{C/N} \approx N\tau\log\frac{\Delta_0 N}{C}$$

For typical fine-tuning ($\Delta_0 \sim 0.5$, $C/N \sim 0.05$, $\tau \sim 10$ steps): $B_\text{crossover} \sim 10N\log(10N/C)$. With $N = 8$: $B_\text{crossover} \sim 200$ steps. With $N = 32$: $\sim 800$ steps.

In practice we operate well below $B_\text{crossover}$ for larger $N$ — so **$N$ converges faster than $2N$ in all our experiments**, and the asymptotic floor difference never materializes within the training budget.

---

## Core Assumptions: What the Theory Actually Requires

The N-cancellation and degeneracy results rest on different assumptions. Being explicit about which:

### N-cancellation assumptions

| Assumption | How critical | Where it breaks |
|---|---|---|
| $\eta^*(N) = N/d$ (step size scales with N) | **Critical** — the cancellation is a direct consequence of this scaling | Any fixed-$\eta$ experiment; breaks cancellation, makes smaller $N$ strictly better |
| Fixed budget $B = N \times T$ | Critical — defines the tradeoff | If $T$ is held fixed instead, $2N$ straightforwardly gives $2\times$ progress |
| Gradient approximately constant over budget | Moderate — requires slow-moving landscape | Sharp loss transitions; large $T$ |
| $L$-smooth $f$ | Needed for the descent lemma (prerequisite), not for the cancellation itself | Binary reward (discontinuous); only formal for CE reward |

### Degeneracy / $N_\text{min}$ assumptions

| Assumption | How critical | Where it breaks |
|---|---|---|
| i.i.d. Bernoulli examples with shared $p_0$ | **Critical** for the closed-form $P(A=0)$ formula | Heterogeneous batches (easy/hard mix); effective $P(A=0)$ is a weighted average |
| Homogeneous $p_0$ across the batch | Moderate | Batches with varying difficulty; some examples always flip, some never do |
| Small $\sigma$ ($p^+ \approx p^- \approx p_0$) | Moderate | Large $\sigma$; perturbation shifts the mean accuracy |
| Normal approximation to Binomial | Mild — requires $B \geq 8$ | Very small batches ($B \leq 4$) |

### The $L$-smoothness question specifically

$L$-smoothness is needed for the descent lemma, not the N-cancellation. But more importantly: $L$-smoothness holds for CE reward (log-probability is smooth in parameters) but is **technically violated for binary reward** (correct/incorrect is a step function of the logits). The descent lemma doesn't formally apply to our primary reward signal. The theory is built for CE reward and is applied to binary reward as an approximation.

---

## $N_\text{min}$ Is Dynamic: $p_0$ Changes During Training

The degeneracy formula $N_\text{min} = \lceil \log\alpha / \log P(A=0) \rceil$ has a hidden assumption: $p_0$ is fixed. In reality it is not.

$p_0$ rises throughout training, and $P(A=0) = 1/\sqrt{2\pi B p_0(1-p_0)}$ is not monotone in $p_0$ — it is minimized at $p_0 = 0.5$ and grows toward both extremes. So $N_\text{min}$ follows a **U-shape** over training:

| Stage | $p_0$ | $P(A=0)$ (B=16) | $N_\text{min}$ |
|---|---|---|---|
| Initialization (base model) | $\approx 0.01$ | $\approx 0.89$ | $\approx 27$ |
| Early fine-tuning | $\approx 0.3$ | $\approx 0.15$ | $2$ |
| Mid training | $\approx 0.5$ | $\approx 0.14$ | $2$ |
| Near convergence | $\approx 0.9$ | $\approx 0.33$ | $4$ |

$N_\text{min}$ is highest at the very start (base model is lost) and near convergence (model is confident). The middle of training is the easiest regime for degeneracy.

This means the $N = 30$ requirement seen empirically reflects the initialization regime — exactly where $p_0 \approx 0.01$ and $N_\text{min} \approx 29$. Once training is underway and $p_0$ rises above 0.2, $N = 4$–$8$ would be sufficient for degeneracy control.

### Empirically measuring $P(A=0)$

Rather than computing $P(A=0)$ from $p_0$ via the formula (which requires the homogeneous batch approximation), it can be measured directly:

1. Sample $K = 200$–$500$ perturbation pairs $\{\varepsilon_i\}$
2. For each pair, evaluate $r(\theta + \sigma\varepsilon_i, \mathcal{B})$ and $r(\theta - \sigma\varepsilon_i, \mathcal{B})$
3. $\hat{P}(A=0) = \frac{1}{K}|\{i : r^+ = r^-\}|$
4. Plug into $N_\text{min} = \lceil \log\alpha / \log\hat{P}(A=0) \rceil$

This **pre-training degeneracy probe** costs a single evaluation pass over the base model (no gradient, no training) and gives an empirical $N_\text{min}$ that automatically accounts for heterogeneous batches, non-standard $\sigma$, and the actual task difficulty — bypassing all the approximations in the formula. Running it once before a pop-scaling sweep would replace the current guesswork about population size with a principled lower bound.

### Conservative lower bound that matches the empirical result

With $p_0 < 0.1$ (base model almost never produces a correct label token on a classification prompt) and $P(A=0) = 0.9$ ($\alpha = 0.05$):

$$N_\text{min} = \left\lceil \frac{\log 0.05}{\log 0.90} \right\rceil = \left\lceil \frac{-2.996}{-0.105} \right\rceil = 29$$

This matches the empirical $N \approx 30$ exactly. The formula gives the correct order of magnitude under realistic assumptions about base model quality — the theory and the experiments are consistent.

---

## Why ES Was Thought to Fail in High Dimensions — and Why It Doesn't

### The Classical Fear: Variance Blowup

The classical argument against ES in high dimensions comes from the variance of the gradient estimator. For $n$ antithetic perturbation pairs, the MeZO Lemma 2 result gives:

$$\mathbb{E}\!\left[\|\hat{\nabla} f\|^2\right] = \frac{d + n - 1}{n} \cdot \mathbb{E}\!\left[\|\nabla f\|^2\right]$$

At $n = 1$ (one perturbation direction), the gradient second moment is **$d\times$ inflated** relative to the true gradient. To keep updates stable, the learning rate must shrink proportionally:

$$\eta_\text{ZO} = \frac{n}{d + n - 1} \cdot \eta_\text{SGD} \approx \frac{1}{d} \cdot \eta_\text{SGD}$$

For $d = 350\text{M}$, this means $350\text{M}\times$ more steps than SGD. For $d = 7\text{B}$, that is $7\text{B}\times$ more steps — seemingly intractable.

This is why classical ZO literature [Nesterov & Spokoiny 2017, Duchi et al. 2015] concluded ZO methods are catastrophically slow for high-dimensional problems.

### Why This Is Wrong for Pretrained LLMs

The $d$-blowup assumes the gradient is spread across all $d$ dimensions equally. For pretrained LLMs, this assumption fails completely. The loss landscape near a good pretrained checkpoint is nearly flat in most directions — only $r \ll d$ directions matter, where $r$ is the effective rank of the Hessian (MeZO Theorem 1 / Lemma 3):

$$T = O\!\left(\left(\frac{r}{n} + 1\right) \cdot \left(\frac{\ell}{\mu} + \frac{\ell\alpha}{\mu^2 B}\right) \cdot \log\frac{f(\theta_0) - f^*}{\varepsilon}\right)$$

Convergence scales with $r/n$, not $d/n$. Empirically $r \sim O(100)$ for LLM fine-tuning tasks, so the actual blowup is $\sim 100\times$ rather than $350\text{M}\times$ — entirely tractable.

The classical analysis assumed an arbitrary function in $\mathbb{R}^d$ with no structure. Pretrained LLMs are maximally structured: their loss landscapes are nearly flat in all but a low-dimensional subspace. That structure is what makes ES viable.

---

## Why ES Fails at Small N — The SNR Argument

### The Setup

For each perturbation seed $\varepsilon$, ES computes an advantage:

$$A = r(\theta + \sigma\varepsilon,\, \mathcal{B}) - r(\theta - \sigma\varepsilon,\, \mathcal{B})$$

and updates: $\theta \leftarrow \theta + \frac{\eta}{\sigma} \cdot \frac{1}{N}\sum_{i=1}^N A_i \varepsilon_i$.

For this update to improve $f$, the gradient estimate $\hat{g} = \frac{1}{N\sigma}\sum_i A_i\varepsilon_i$ must be sufficiently aligned with the true gradient $\nabla f$. This requires high **signal-to-noise ratio** (SNR) in each advantage $A_i$.

### Binary Accuracy Reward: Low SNR

With binary accuracy reward, the advantage decomposes as:

$$A_\text{acc} = \underbrace{2\sigma \langle \nabla f, \varepsilon \rangle}_{\text{true signal}} + \underbrace{\xi_\text{batch}}_{\text{batch noise}}$$

where $\xi_\text{batch}$ is the noise from which $B$ examples happened to be in the batch. The **true signal** comes only from examples that **flip** their correctness between $+\sigma\varepsilon$ and $-\sigma\varepsilon$. The probability of a single example flipping is:

$$P(\text{flip}_i) \approx 2\sigma\,|\langle \nabla p_i(\theta), \varepsilon \rangle|$$

For small $\sigma$ (e.g., $10^{-3}$), very few examples flip. If $k$ examples flip in a batch of $B$:

$$|A_\text{acc}| = \frac{2k}{B} \in \left\{0,\, \frac{2}{B},\, \frac{4}{B},\, \ldots\right\}$$

The batch noise variance is:

$$\text{Var}[\xi_\text{batch}] \approx \frac{2(1 - p_0^2)}{B}$$

For $p_0 = 0.8$, $B = 16$: $\text{std}[\xi_\text{batch}] \approx 0.21$.

With typically only 1 example flipping, the signal is $|A_\text{acc}| \approx 2/16 = 0.125$. So:

$$\text{SNR}_\text{acc} \approx \frac{0.125}{0.21} \approx 0.6$$

**Below 1** — the noise exceeds the signal in a single seed. The gradient estimate is dominated by which batch happened to be drawn, not by the true reward landscape.

### Cross-Entropy Reward: High SNR

With CE reward, every example contributes a continuous signal regardless of whether it flips:

$$A_\text{CE} = \frac{1}{B}\sum_{i=1}^B \left[\log p(\theta+\sigma\varepsilon, x_i) - \log p(\theta-\sigma\varepsilon, x_i)\right] \approx \frac{2\sigma}{B}\sum_{i=1}^B \langle \nabla \log p_i, \varepsilon \rangle$$

This is $2\sigma \langle g_\text{CE}, \varepsilon \rangle$ where $g_\text{CE} = \frac{1}{B}\sum_i \nabla \log p_i$ is the batch CE gradient. Crucially:

- All $B$ examples contribute, not just the $k \ll B$ that flip
- The signal scales as $2\sigma \|g_\text{CE}\| / \sqrt{d}$ (projection onto random direction)
- The batch noise is much smaller because log-probabilities vary smoothly

$$\text{SNR}_\text{CE} \gg \text{SNR}_\text{acc}$$

This is the formal reason MeZO converges with $N = 1$: every SPSA pair yields a signal aligned with the true CE gradient, without the discretization noise from counting flipped examples.

### Why Small N Causes Performance to Decrease, Not Just Stagnate

With $N$ small and SNR below 1, the gradient estimate $\hat{g}$ is essentially random noise. The expected loss after one step is:

$$\mathbb{E}[f(\theta - \eta\hat{g})] \approx f(\theta) \underbrace{- \eta\|\nabla f\|^2}_{\text{improvement from signal}} + \underbrace{\frac{\eta^2 L}{2}\|\hat{g}\|^2}_{\text{harm from noise}}$$

When $\|\hat{g}\|^2 \gg \|\nabla f\|^2$ (noise dominates), the harm term wins and the loss **increases**. The model drifts away from the pretrained initialization in random directions.

This is not a flat random walk — it is a systematically harmful one. The pretrained model sits near a good region of a roughly convex loss landscape. Most random directions from that point are uphill. A noise-dominated update with fixed learning rate $\eta$ causes:

$$\mathbb{E}[f(\theta_t)] \approx f(\theta_0) + t \cdot \frac{\eta^2 L}{2}\|\hat{g}_\text{noise}\|^2$$

Performance degrades monotonically until the learning rate decays or the model reaches a worse basin.

This explains the empirical observation that ES with $N \in \{1, 2, 4\}$ and binary reward shows **performance below baseline** — not just failure to improve, but active degradation. The ES Scale paper (arXiv:2601.20861) independently confirmed this: ES updates have much larger L2 norms than GRPO, causing the model to drift far from its initialization (catastrophic forgetting of prior capabilities).

### Summary

| N regime | Binary reward | CE reward |
|---|---|---|
| $N = 1$ | SNR $< 1$, noise-dominated update, performance degrades | SNR $> 1$, signal-aligned update, makes progress |
| $N = 4$–$8$ | Signal starts to emerge from noise averaging | Works well |
| $N \geq 30$ | Signal reliably dominates noise | Overkill — wastes compute |
| $N \to r$ | Optimal: $N^* \leq r$ from effective rank bound | Same upper bound |

The minimum population size for binary reward is set by the SNR condition — you need enough seeds to average out the batch noise and recover a reliable gradient direction. For CE reward, even $N = 1$ gives a reliable direction because the signal is continuous and uses every example, not just the rare flippers.
