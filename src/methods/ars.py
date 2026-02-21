"""Augmented Random Search (ARS)."""

from __future__ import annotations

import numpy as np

from src.envs.base import rollout, eval_policy
from src.methods.base import MethodResult
from src.policy import LinearPolicy, RunningNorm


def _make_policy_fn(W_flat, env, running_norm):
    """Create a policy function from a flat weight vector."""
    W = W_flat.reshape(env.action_dim, env.obs_dim)

    def policy_fn(obs):
        return W @ obs

    return policy_fn


def _build_eval_env_if_needed(config, train_env):
    """Use standard-reward MuJoCo env for eval when training strips survival bonus."""
    if (
        getattr(config.env, "name", None) in {"hopper", "walker2d", "ant", "humanoid"}
        and getattr(config.env, "remove_survival_bonus", False)
    ):
        from src.envs.mujoco import MuJoCoEnv
        return MuJoCoEnv(
            task_name=config.env.name,
            remove_survival_bonus=False,
            max_steps=config.max_steps,
        )
    return train_env


def run_ars(config, env, seed: int) -> list[MethodResult]:
    """Run ARS training, return per-iteration results."""
    mc = config.method
    rng = np.random.default_rng(seed)

    policy = LinearPolicy(env.obs_dim, env.action_dim)
    running_norm = RunningNorm(env.obs_dim) if mc.use_state_norm else None

    num_iters = config.num_iters()
    episodes_consumed = 0
    results = []
    eval_env = _build_eval_env_if_needed(config, env)
    owns_eval_env = eval_env is not env

    for t in range(num_iters):
        # Sample perturbation directions
        deltas = [rng.standard_normal(policy.theta.shape) for _ in range(mc.N)]

        rewards_pos = []
        rewards_neg = []

        for k in range(mc.N):
            # Positive perturbation
            theta_pos = policy.theta + mc.sigma * deltas[k]
            fn_pos = _make_policy_fn(theta_pos, env, running_norm)
            ep_seed = seed * 1_000_000 + t * 1000 + 2 * k
            r_pos, _ = rollout(env, fn_pos, ep_seed, config.max_steps, running_norm)
            rewards_pos.append(r_pos)

            # Negative perturbation
            theta_neg = policy.theta - mc.sigma * deltas[k]
            fn_neg = _make_policy_fn(theta_neg, env, running_norm)
            ep_seed_neg = seed * 1_000_000 + t * 1000 + 2 * k + 1
            r_neg, _ = rollout(env, fn_neg, ep_seed_neg, config.max_steps, running_norm)
            rewards_neg.append(r_neg)

        episodes_consumed += 2 * mc.N

        # Top-b selection by max(r_pos, r_neg)
        max_rewards = [max(rp, rn) for rp, rn in zip(rewards_pos, rewards_neg)]
        top_idx = np.argsort(max_rewards)[-mc.b:]

        # Update
        step = np.zeros_like(policy.theta)
        for i in top_idx:
            step += (rewards_pos[i] - rewards_neg[i]) * deltas[i]

        if mc.reward_norm:
            # Normalize step by std of rewards used (ARS V2/V2-t; paper eq. 1)
            rewards_used = [rewards_pos[i] for i in top_idx] + [rewards_neg[i] for i in top_idx]
            sigma_R = max(np.std(rewards_used), 1e-8)
            policy.theta = policy.theta + (mc.lr / (mc.b * sigma_R)) * step
        else:
            # No reward normalization (ARS V1/V1-t / BRS)
            policy.theta = policy.theta + (mc.lr / mc.b) * step

        # Eval
        eval_ret = None
        if (t + 1) % config.eval_every_iters == 0 or t == num_iters - 1:
            eval_seed = seed * 1_000_000 + 999_000 + t
            fn_eval = _make_policy_fn(policy.theta, eval_env, running_norm)
            eval_ret = eval_policy(
                eval_env, fn_eval, eval_seed, config.eval_episodes, config.max_steps, running_norm
            )

        all_train = rewards_pos + rewards_neg
        results.append(
            MethodResult(
                iteration=t,
                episodes_consumed=episodes_consumed,
                train_returns=all_train,
                eval_return=eval_ret,
            )
        )

    if owns_eval_env:
        eval_env.close()
    return results
