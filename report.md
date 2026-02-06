# Project Report

## 1. Problem Statement

### What are you optimizing? (Be specific)

We are optimizing hyperparameters for an open-weight Qwen model to play Wordle under a limited compute budget.

Concretely, the project is structured as two stages:

- **Baseline**: train Qwen for Wordle using a policy-gradient method (e.g., REINFORCE / PPO-style variants) and measure **training time** and **Wordle performance** under a fixed compute budget.
- **Main experiment**: train Qwen for Wordle using **evolutionary strategies (ES)**, and optimize ES hyperparameters that maximize performance given the same compute budget.

In the ES stage, the object we optimize is a hyperparameter vector \(h\) (that consists of information such as perturbation scale, population size, learning rate, reward normalization settings, number of rollouts per fitness evaluation, etc.). The objective is the final evaluation performance after training with ES configured by \(h\), subject to the compute budget.

### Why does this problem matter?

Conventional policy gradient reinforcement learning methods require substantial memory resources, particularly when training large 
models. Evolutionary strategies offer a promising alternative with significantly improved memory efficiency, as they do not require 
storing gradients or maintaining a replay buffer. However, tuning LLMs using zeroth-order optimization methods presents unique 
challenges compared to first-order methods, making effective hyperparameter selection critical for achieving competitive performance.

### How will you measure success?

We will evaluate success with metrics that directly reflect “can it play Wordle well under a budget?”:

- **Task performance**: Wordle win-rate, average number of guesses (conditional on win), and/or average episode reward on a fixed evaluation set of target words.
- **Compute efficiency**: wall-clock training time and total environment interactions / rollouts used to reach a given performance threshold.
- **Fair comparison**: best achieved performance **at equal compute** between (a) policy-gradient baseline and (b) ES with hyperparameters chosen by the outer loop.

### What are your constraints?

Our primary constraints are **limited training time** and a **fixed compute budget** (GPU hours). Because both policy gradients and ES are sensitive to randomness, we will also constrain experiments to a small number of seeds and prioritize methods that can be compared fairly under identical budgets.

### What data do you need?

We will use a Wordle environment (on-the-fly episodes) for training and evaluation. The “dataset” is primarily:

- **Training data sets**: generated interactively by the environment.
- **Testing data set**: a fixed list of target words held out from any tuning decisions, used to produce stable comparisons across methods.

### What could go wrong?

Key risks include inconsistent experimental conditions across runs, which could invalidate comparisons. We must ensure that all experiments adhere to the same constraints, use identical model architectures, and maintain consistent evaluation protocols. Additionally, the stochastic nature of evolutionary strategies may require multiple runs to obtain reliable performance estimates.

## 2. Technical Approach

### Mathematical formulation (objective function, constraints)

The optimization problem can be formulated as maximizing the expected reward over policy parameters $\theta$:

$$
\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]
$$

where $R(\tau)$ is the cumulative reward for trajectory $\tau$. Evolutionary strategies approximate the gradient using finite differences:

$$
\nabla_{\theta} \mathbb{E}[R] \approx \frac{1}{\sigma^2} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [\epsilon \cdot R(\theta + \sigma\epsilon)]
$$

where $\sigma$ is the perturbation scale and $\epsilon$ is a random perturbation vector. The key hyperparameters to optimize include the perturbation scale $\sigma$, population size $N$, learning rate $\alpha$, and potentially covariance matrix adaptation parameters. Constraints include memory budget $M_{\text{max}}$ and total compute budget $C_{\text{max}}$.

### Algorithm/approach choice and justification

- **Baseline**: a straightforward policy-gradient approach is used first to establish “what can we do with our budget?” and to produce a reference learning curve and final performance.
- **ES training**: we will start with a simple, scalable ES variant (e.g., NES with antithetic sampling and reward normalization) because it is easy to implement and robust to sparse rewards when tuned.
- **Hyperparameter optimization**: because ES is sensitive to hyperparameters and evaluation noise, we will run an outer-loop search to select \(h\) under the same overall budget.

### PyTorch implementation strategy

Our PyTorch implementation will leverage parameter perturbation techniques where we sample perturbations $\epsilon_i \sim \mathcal{N}(0,I)$ and evaluate policies at $\theta + \sigma\epsilon_i$. We will use PyTorch's parameter cloning and manipulation capabilities to efficiently generate perturbed parameter sets without requiring gradient computation. We will implement batched evaluations where possible to maximize GPU utilization while respecting memory constraints. The optimization update will follow:

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{N} \sum_{i} (\epsilon_i \cdot R_i)
$$

where $R_i$ is the reward for perturbation $i$.

### Validation methods

Validation is primarily comparative:

- **Budget-matched comparison**: compare ES (with tuned hyperparameters) vs policy-gradient baseline at equal compute.
- **Seeded evaluation**: use a small, fixed set of seeds and report mean/variance where feasible.
- **Learning curves**: track performance vs time/rollouts to show training efficiency, not just final score.

### Resource requirements and constraints

The main constraint is **total compute budget**. We will explicitly allocate budget across:

- **Baseline run** (policy gradients): one or more runs to obtain a stable baseline curve.
- **Outer-loop hyperparameter search** (ES): a limited number of trials, each with a capped per-trial training budget.

We will implement checkpointing and logging so that:

- training can be resumed without wasting budget
- comparisons are reproducible (fixed evaluation set, logged configs, and versioned code)

## 3. Initial Results

Current status (to be filled as experiments run):

- Evidence the Wordle environment and training loop runs
- Establish baseline policy-gradient training time and evaluation performance under the budget

## 4. Next Steps

- Immediate improvements needed
- Technical challenges to address
- Questions you need help with
### Alternative approaches to try
If the results are inconclusive, may try another environment than Wordle. However, we note that hyperparamaters are vastly different depending on the landscape, meaning that our optimized values (and possibly tuning methods) will change when we change our objective.
- What you've learned so far
