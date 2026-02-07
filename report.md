# Project Report

**Team:** Sung Cho, Maxwell DeLorenzo, Gyubin Han

## 1. Problem Statement

### What are you optimizing?

We study optimization and tuning strategies for zeroth-order reinforcement learning (RL) methods applied to large language models (LLMs). Concretely, we aim to understand how policy gradient (PG) methods and evolutionary strategies (ES) behave when applied to LLM-driven RL environments, under realistic constraints on observability, memory, and controllability.

Rather than directly optimizing model weights, we initially focus on prompt-level policies (e.g., selecting or weighting prompt modules) as a controllable policy space compatible with black-box evaluation environments.

### Why does this problem matter?

Many modern RL environments for LLMs, including hosted or agent-based platforms, do not expose gradients, intermediate states, or step-level rewards. In such settings:

- First-order methods are difficult or impossible to apply directly.
- Zeroth-order methods (e.g., ES) become attractive alternatives.
- Even policy gradient methods must often be reformulated over non-standard policy spaces (e.g., prompts rather than weights).

Understanding how these optimization methods behave under constrained, black-box evaluation pipelines is therefore crucial for practical post-training and agent optimization of LLMs.

**Why Wordle + Prime Intellect?**

We choose Wordle-style environments hosted via Prime Intellect for three reasons:

1. **Black-box reward structure:** rewards are computed externally via Prime's evaluator, closely matching real-world post-training settings.
2. **Sequential decision-making with sparse feedback:** Wordle highlights credit assignment challenges common in LLM-based RL.
3. **Infrastructure realism:** Prime Intellect reflects the tooling and constraints encountered in modern LLM research platforms.

Because Prime environments do not expose model internals or gradients, this setting naturally motivates zeroth-order optimization and prompt-based control.

**Why prompt policies instead of weight policies?**

In the Prime Intellect + Wordle setting:

- Direct weight updates are not supported in local evaluation mode.
- The environment operates through text-based interaction and rubric-level rewards.
- Prompt structure is one of the few controllable and reproducible knobs.

As a result, we define policies over prompt modules (e.g., formatting constraints, tracking instructions, information-gain heuristics). This choice allows us to:

- Implement both PG and ES in a comparable way,
- Control the policy space explicitly,
- Remain faithful to the black-box nature of the environment.

Whether prompt policies are sufficient or merely a proxy for weight-level learning is an open question we plan to revisit.

### How will you measure success?

At this stage, success is defined by:

- **Correctness:** PG and ES updates produce non-trivial policy changes and non-zero rewards.
- **Comparability:** PG and ES can be evaluated under matched evaluation budgets.
- **Qualitative optimization behavior:** differences in variance, stability, and sample efficiency are observable.

Later stages will focus more heavily on performance metrics (e.g., success rate, convergence trends).

### Constraints

- Fixed evaluation budget (Prime CLI calls are expensive).
- No access to gradients, environment internals, or step-level rewards.
- Linux-only Prime tooling (requiring WSL on Windows).
- Practical time constraints for debugging and experimentation.

### What data do you need?

We rely entirely on on-the-fly interaction data generated through Prime Intellect’s evaluation environments. Each training signal consists of complete episode trajectories and scalar rewards returned by the Prime evaluator, without access to intermediate states or gradients.

### What could go wrong?

Because rewards are sparse, delayed, and externally computed, optimization signals may be noisy or misleading at small scales. Additionally, prompt-level policies may fail to capture deeper model behaviors, limiting the interpretability and generalizability of observed optimization dynamics.


## 2. Technical Approach

### Objective formulation

We consider the standard RL objective:

$$\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)],$$

where $R(\tau)$ is the scalar reward returned by the Prime evaluator for a full episode (trajectory).

Crucially, $R(\tau)$ is:

- Sparse
- Delayed
- Externally computed

This motivates optimization methods that rely only on function evaluations, not gradients.

### Policy Gradient (REINFORCE)

For PG, we define a Bernoulli policy over prompt modules:

$$a \sim \pi_{\theta}(a), \quad a \in \{0,1\}^K$$

and update parameters using REINFORCE:

$$\theta \leftarrow \theta + \alpha (R - b) \nabla_{\theta} \log \pi_{\theta}(a),$$

where $b$ is a baseline to reduce variance.

Each PG update consists of multiple Prime evaluations corresponding to sampled prompt configurations.

### Evolutionary Strategies (ES)

For ES, we define continuous prompt parameters $\theta \in \mathbb{R}^K$ and apply Gaussian perturbations:

$$\epsilon_i \sim \mathcal{N}(0, I), \quad R_i = R(\theta + \sigma \epsilon_i)$$

with gradient estimate:

$$\nabla_{\theta} J \approx \frac{1}{N} \sum_i (R_i - b) \epsilon_i.$$

ES treats the entire evaluation pipeline as a black-box reward oracle, making it well-suited to Prime's interface.

### Evaluation pipeline

Both PG and ES use the same evaluation mechanism:

1. `prime eval run` via CLI
2. Results saved to disk
3. Scalar rewards extracted from `results.json(l)`
4. Aggregation performed externally

This ensures methodological parity between PG and ES.

## 3. Initial Results

At this stage, our results focus on system correctness and feasibility, rather than final performance.

### Evidence the implementation works

- Prime CLI evaluation successfully returns non-zero rewards.
- Policy Gradient updates modify Bernoulli prompt logits.
- Evolutionary Strategies produce non-trivial gradient estimates and parameter updates.
- PG and ES can be run under matched evaluation budgets.

### Engineering outcomes

- A robust reward-extraction pipeline was implemented despite varying Prime output formats.
- PG and ES share a common black-box evaluator interface.
- Prompt-based policies provide a workable control surface under Prime's constraints.

### Results interpretation

Quantitative performance comparisons and plots will be added after notebook experiments are finalized.

## 4. Next Steps

### Immediate next steps

- Run short PG vs ES experiments with matched evaluation budgets.
- Plot reward trajectories and variance over update steps.
- Compare qualitative optimization behavior.

### Technical challenges encountered

- Prime Intellect setup required Linux-only tooling; development was performed via WSL on Windows, introducing tooling limitations.
- Prime evaluation outputs vary in directory structure and file formats, requiring non-trivial debugging.
- TextArena/Wordle environments provide limited controllable knobs despite rich interaction dynamics.
- Significant debugging effort was required to align evaluation artifacts with optimization loops.

### Questions we need help with

- Is prompt-level policy optimization an adequate proxy for studying LLM post-training dynamics?
- How should evaluation budgets be normalized when comparing PG and ES?
- To what extent do prompt policies capture meaningful optimization behavior compared to weight-level updates?

### Limitations

- We cannot afford large-scale or long-horizon training due to evaluation costs.
- Model choice is constrained by API pricing and availability.
- Hosted environments limit observability and direct control.
- Infrastructure constraints (Linux-only tooling) complicate development workflows.

### What we have learned so far

- Zeroth-order methods integrate naturally with black-box LLM evaluation pipelines.
- PG and ES behave qualitatively differently even in small-scale prompt-policy settings.
- Engineering complexity is a central consideration in modern LLM RL research.