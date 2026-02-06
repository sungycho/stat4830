# Project Report

## 1. Problem Statement

### What are you optimizing? (Be specific)

We are optimizing hyperparameter tuning processes for evolutionary strategies (ES) applied to reinforcement learning tasks with large language models (LLMs). Specifically, we aim to develop effective and replicable tuning procedures for zeroth-order optimization methods that can efficiently train LLMs on different RL environments.

### Why does this problem matter?

Conventional policy gradient reinforcement learning methods require substantial memory resources, particularly when training large models. Evolutionary strategies offer a promising alternative with significantly improved memory efficiency, as they do not require storing gradients or maintaining a replay buffer. However, tuning LLMs using zeroth-order optimization methods presents unique challenges compared to first-order methods, making effective hyperparameter selection critical for achieving competitive performance.

### How will you measure success?

We will evaluate success using three primary metrics: (1) model accuracy on the target task (e.g., Wordle solving performance), (2) training time required to reach convergence, and (3) peak memory usage during training. Ideally, we would achieve comparable or better accuracy than policy gradient baselines while significantly reducing memory consumption.

### What are your constraints?

Our primary constraints include limited training time and a fixed compute budget. Additionally, we must ensure that experiments are reproducible and that all comparisons are made under equivalent computational and environmental conditions.

### What data do you need?

We will primarily use environments provided by Prime Intellect, which enable on-the-fly data generation for training and evaluation. Specifically, we will begin by generating baseline comparison data by training an open-weight LLM (e.g., Qwen) using policy gradient methods to solve Wordle, measuring both accuracy and memory usage to establish performance benchmarks.

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

We will explore several evolutionary strategy variants, including Natural Evolution Strategies (NES) and potentially CMA-ES for covariance adaptation. NES is chosen for its simplicity and effectiveness in high-dimensional spaces, while CMA-ES offers adaptive covariance estimation that can improve convergence. The zeroth-order nature of these methods eliminates the need for backpropagation through the environment, significantly reducing memory requirements compared to policy gradient methods. We will implement a hyperparameter search strategy (e.g., grid search or Bayesian optimization) to systematically explore the hyperparameter space.

### PyTorch implementation strategy

Our PyTorch implementation will leverage parameter perturbation techniques where we sample perturbations $\epsilon_i \sim \mathcal{N}(0,I)$ and evaluate policies at $\theta + \sigma\epsilon_i$. We will use PyTorch's parameter cloning and manipulation capabilities to efficiently generate perturbed parameter sets without requiring gradient computation. The fitness evaluation will involve forward passes through the LLM policy network on sampled trajectories. We will implement batched evaluations where possible to maximize GPU utilization while respecting memory constraints. The optimization update will follow:

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{N} \sum_{i} (\epsilon_i \cdot R_i)
$$

where $R_i$ is the reward for perturbation $i$.

### Validation methods

Validation will be performed through systematic comparison against policy gradient baselines (e.g., PPO or REINFORCE) on identical tasks and model architectures. We will use cross-validation across multiple random seeds to ensure statistical significance. Performance will be evaluated on held-out test sets, and we will track convergence curves to compare training efficiency. Memory profiling will be conducted using PyTorch's memory tracking utilities to measure peak memory consumption during training.

### Resource requirements and constraints

The implementation must operate within memory constraints suitable for training LLMs (targeting <50% of baseline policy gradient memory usage). We will utilize GPU resources efficiently through batched evaluations and careful management of parameter copies. The compute budget will be allocated across hyperparameter search iterations, with each iteration requiring $N$ forward passes (where $N$ is the population size) plus reward computation. We will implement checkpointing to enable resumption of long-running experiments and to manage compute budget effectively.

## 3. Initial Results

- Evidence your implementation works
- Basic performance metrics
- Test case results
- Current limitations
- Resource usage measurements
- Unexpected challenges

## 4. Next Steps

- Immediate improvements needed
- Technical challenges to address
- Questions you need help with
- Alternative approaches to try
- What you've learned so far
