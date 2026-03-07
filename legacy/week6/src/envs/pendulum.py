"""Pendulum environment wrapper with shared interface."""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class PendulumEnv:
    """Thin wrapper around gymnasium Pendulum-v1 with optional obs noise."""

    def __init__(self, obs_noise_std: float = 0.0, max_steps: int = 200):
        self._env = gym.make("Pendulum-v1")
        self.obs_dim = 3
        self.action_dim = 1
        self.action_low = np.array([-2.0])
        self.action_high = np.array([2.0])
        self._obs_noise_std = obs_noise_std
        self._max_steps = max_steps
        self._noise_rng = None
        self._t = 0

    def reset(self, seed: int = 0) -> np.ndarray:
        obs, _ = self._env.reset(seed=int(seed))
        self._noise_rng = np.random.default_rng(int(seed) + 1_000_000)
        self._t = 0
        if self._obs_noise_std > 0:
            obs = obs + self._noise_rng.standard_normal(self.obs_dim) * self._obs_noise_std
        return obs.astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        action = np.asarray(action, dtype=np.float64).ravel()
        obs, reward, terminated, truncated, _ = self._env.step(action)
        self._t += 1
        if self._obs_noise_std > 0:
            obs = obs + self._noise_rng.standard_normal(self.obs_dim) * self._obs_noise_std
        done = terminated or truncated or self._t >= self._max_steps
        return obs.astype(np.float64), float(reward), done

    def close(self):
        self._env.close()
