"""Shared rollout interface for all environments and methods."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.policy import RunningNorm


def rollout(
    env,
    policy_fn,
    seed: int,
    max_steps: int,
    running_norm: RunningNorm | None = None,
) -> tuple[float, int]:
    """Run a single episode, return (total_return, steps).

    This is the canonical rollout used by ALL methods.
    """
    obs = env.reset(seed=int(seed))
    total_return = 0.0
    for t in range(max_steps):
        if running_norm is not None:
            running_norm.update(obs)
            obs_input = running_norm.normalize(obs)
        else:
            obs_input = obs

        action = policy_fn(obs_input)
        action = np.clip(action, env.action_low, env.action_high)
        obs, reward, done = env.step(action)
        total_return += reward
        if done:
            return total_return, t + 1
    return total_return, max_steps


def eval_policy(
    env,
    policy_fn,
    base_seed: int,
    episodes: int,
    max_steps: int,
    running_norm: RunningNorm | None = None,
) -> float:
    """Evaluate a policy over multiple episodes, return mean return.

    Eval seeds: base_seed + 10_000 + k to avoid collision with training seeds.
    Running norm is NOT updated during eval (read-only).
    """
    returns = []
    for k in range(episodes):
        ep_seed = base_seed + 10_000 + k

        # Freeze running norm during eval
        class FrozenNorm:
            def __init__(self, rn):
                self._rn = rn

            def update(self, x):
                pass  # no-op during eval

            def normalize(self, x):
                return self._rn.normalize(x)

        frozen = FrozenNorm(running_norm) if running_norm is not None else None
        ret, _ = rollout(env, policy_fn, ep_seed, max_steps, frozen)
        returns.append(ret)
    return float(np.mean(returns))
