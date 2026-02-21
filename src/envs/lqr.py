"""Discrete-time LQR environment."""

from __future__ import annotations

import numpy as np


class LQREnv:
    """Discrete-time LQR: x_{t+1} = Ax + Bu + noise, r = -(x'Qx + u'Ru).

    A and B are generated from system_seed with spectral radius < 1.
    Q = I, R = I.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        horizon: int = 200,
        noise_std: float = 0.1,
        system_seed: int = 42,
    ):
        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.noise_std = noise_std
        self.action_low = np.full(action_dim, -10.0)
        self.action_high = np.full(action_dim, 10.0)

        # Generate stable system matrices
        rng = np.random.default_rng(system_seed)
        A = rng.standard_normal((state_dim, state_dim))
        eigvals = np.abs(np.linalg.eigvals(A))
        spectral_radius = np.max(eigvals)
        A = A / (max(1.0, spectral_radius) + 0.1)
        self.A = A
        self.B = rng.standard_normal((state_dim, action_dim))
        self.Q = np.eye(state_dim)
        self.R = np.eye(action_dim)

        self._rng = None
        self._state = None
        self._t = 0

    def reset(self, seed: int = 0) -> np.ndarray:
        self._rng = np.random.default_rng(int(seed))
        self._state = self._rng.standard_normal(self.obs_dim) * 0.5
        self._t = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        action = np.asarray(action, dtype=np.float64).ravel()
        x = self._state
        cost = float(x @ self.Q @ x + action @ self.R @ action)
        reward = -cost

        noise = self._rng.standard_normal(self.obs_dim) * self.noise_std
        self._state = self.A @ x + self.B @ action + noise
        self._t += 1
        done = self._t >= self.horizon
        return self._state.copy(), reward, done

    def close(self):
        pass
