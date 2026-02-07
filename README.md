# STAT 4830 Project: PG vs ES for LLM Reinforcement Learning

This repository contains a STAT 4830 course project investigating **optimization strategies for large language models (LLMs) in black-box reinforcement learning environments**.  
We focus on comparing **Policy Gradient (PG)** methods and **Evolutionary Strategies (ES)** under realistic constraints imposed by hosted evaluation platforms.

---

## Project Overview

Modern LLM reinforcement learning settings often restrict access to gradients, internal states, or step-level rewards. To study optimization behavior under these constraints, we use **Prime Intellect’s Wordle-style environments** as a black-box reward oracle and define policies over **prompt-level controls** rather than model weights.

Our goal is not to maximize task performance, but to **analyze and compare the qualitative behavior of PG and ES**—including variance, stability, and sample efficiency—when applied to LLM-driven tasks with limited observability.

---

## Methods

- **Policy Gradient (REINFORCE)**  
  Uses a Bernoulli policy over prompt modules and updates parameters via stochastic policy gradients with a baseline.

- **Evolutionary Strategies (ES)**  
  Applies Gaussian perturbations to continuous prompt parameters and estimates gradients via finite differences, treating the evaluator as a black box.

Both methods share the same evaluation pipeline via Prime’s CLI, ensuring fair comparison under matched evaluation budgets.

---

## Code Structure

```text
src/
  pg.py              # Policy Gradient (REINFORCE) implementation
  es.py              # Evolutionary Strategies implementation
  policy_prompt.py   # Prompt policy definitions
  utils_prime.py     # Prime CLI evaluation + reward extraction

environments/
  my_env/            # Wordle-style Prime environment definition

notebooks/
  week4_implementation.ipynb   # Lightweight PG vs ES experiments and plots
