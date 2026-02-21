"""REINFORCE with Gaussian policy (numpy, no torch)."""

from __future__ import annotations

import numpy as np

from src.envs.base import rollout, eval_policy
from src.methods.base import MethodResult
from src.policy import LinearPolicy, RunningNorm


def run_reinforce(config, env, seed: int) -> list[MethodResult]:
    """Run REINFORCE training, return per-iteration results.

    Policy: a ~ N(W @ obs, action_sigma^2 * I), fixed sigma.
    Gradient: sum_t(outer(noise_t, obs_t) / action_sigma^2) per episode.
    """
    mc = config.method
    rng = np.random.default_rng(seed)

    policy = LinearPolicy(env.obs_dim, env.action_dim)
    running_norm = RunningNorm(env.obs_dim) if mc.use_state_norm else None
    action_sigma = mc.action_sigma

    num_iters = config.num_iters()
    K = mc.episodes_per_update
    episodes_consumed = 0
    results = []

    for t in range(num_iters):
        ep_returns = []
        ep_grads = []

        for k in range(K):
            ep_seed = seed * 1_000_000 + t * 1000 + k
            ep_rng = np.random.default_rng(ep_seed + 500_000)

            obs = env.reset(seed=int(ep_seed))
            total_return = 0.0
            grad_accum = np.zeros_like(policy.W)

            for step in range(config.max_steps):
                if running_norm is not None:
                    running_norm.update(obs)
                    obs_input = running_norm.normalize(obs)
                else:
                    obs_input = np.asarray(obs, dtype=np.float64)

                mean_action = policy.W @ obs_input
                noise = ep_rng.standard_normal(env.action_dim) * action_sigma
                action = mean_action + noise
                action_clipped = np.clip(action, env.action_low, env.action_high)

                # Log-prob gradient: d/dW log N(a|Wx, sigma^2 I) = outer(noise, obs) / sigma^2
                grad_accum += np.outer(noise, obs_input) / (action_sigma**2)

                obs, reward, done = env.step(action_clipped)
                total_return += reward
                if done:
                    break

            ep_returns.append(total_return)
            ep_grads.append(grad_accum)

        episodes_consumed += K

        # Baseline
        baseline = np.mean(ep_returns)

        # Policy gradient update
        grad = np.zeros_like(policy.W)
        for k in range(K):
            grad += (ep_returns[k] - baseline) * ep_grads[k]
        grad /= K

        policy.W = policy.W + mc.lr * grad

        # Eval (deterministic: mean action, no noise)
        eval_ret = None
        if (t + 1) % config.eval_every_iters == 0 or t == num_iters - 1:
            eval_seed = seed * 1_000_000 + 999_000 + t

            def deterministic_fn(obs, _w=policy.W.copy()):
                return _w @ obs

            eval_ret = eval_policy(
                env, deterministic_fn, eval_seed, config.eval_episodes,
                config.max_steps, running_norm,
            )

        results.append(
            MethodResult(
                iteration=t,
                episodes_consumed=episodes_consumed,
                train_returns=ep_returns,
                eval_return=eval_ret,
            )
        )

    return results
