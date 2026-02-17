"""Vanilla Evolutionary Strategy (one-sided, no antithetic pairing)."""

from __future__ import annotations

import numpy as np

from benchmark.envs.base import rollout, eval_policy
from benchmark.methods.base import MethodResult
from benchmark.policy import LinearPolicy, RunningNorm


def _make_policy_fn(W_flat, env):
    W = W_flat.reshape(env.action_dim, env.obs_dim)

    def policy_fn(obs):
        return W @ obs

    return policy_fn


def run_vanilla_es(config, env, seed: int) -> list[MethodResult]:
    """Run Vanilla ES training, return per-iteration results."""
    mc = config.method
    rng = np.random.default_rng(seed)

    policy = LinearPolicy(env.obs_dim, env.action_dim)
    running_norm = RunningNorm(env.obs_dim) if mc.use_state_norm else None

    num_iters = config.num_iters()
    episodes_consumed = 0
    results = []

    for t in range(num_iters):
        # Sample N perturbations (one-sided)
        epsilons = [rng.standard_normal(policy.theta.shape) for _ in range(mc.N)]
        rewards = []

        for k in range(mc.N):
            theta_k = policy.theta + mc.sigma * epsilons[k]
            fn_k = _make_policy_fn(theta_k, env)
            ep_seed = seed * 1_000_000 + t * 1000 + k
            r, _ = rollout(env, fn_k, ep_seed, config.max_steps, running_norm)
            rewards.append(r)

        episodes_consumed += mc.N

        # Mean baseline
        baseline = np.mean(rewards)

        # Gradient estimate
        grad = np.zeros_like(policy.theta)
        for k in range(mc.N):
            grad += (rewards[k] - baseline) * epsilons[k]
        grad /= mc.N * mc.sigma

        policy.theta = policy.theta + mc.lr * grad

        # Eval
        eval_ret = None
        if (t + 1) % config.eval_every_iters == 0 or t == num_iters - 1:
            eval_seed = seed * 1_000_000 + 999_000 + t
            fn_eval = _make_policy_fn(policy.theta, env)
            eval_ret = eval_policy(
                env, fn_eval, eval_seed, config.eval_episodes, config.max_steps, running_norm
            )

        results.append(
            MethodResult(
                iteration=t,
                episodes_consumed=episodes_consumed,
                train_returns=rewards,
                eval_return=eval_ret,
            )
        )

    return results
