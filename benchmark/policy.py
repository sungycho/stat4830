"""Shared linear policy and running normalization."""

from __future__ import annotations

import numpy as np


class RunningNorm:
    """Online per-dimension mean/variance using Welford's algorithm."""

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return np.sqrt(self.var + self.eps)

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "count": self.count,
            "mean": self.mean.tolist(),
            "M2": self.M2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> RunningNorm:
        rn = cls(d["dim"], d["eps"])
        rn.count = d["count"]
        rn.mean = np.array(d["mean"], dtype=np.float64)
        rn.M2 = np.array(d["M2"], dtype=np.float64)
        return rn


class LinearPolicy:
    """Linear policy: action = W @ obs.

    W is (action_dim, obs_dim), initialized to zeros.
    `theta` property gives a flat view for optimizer access.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.W = np.zeros((action_dim, obs_dim), dtype=np.float64)

    @property
    def theta(self) -> np.ndarray:
        return self.W.ravel()

    @theta.setter
    def theta(self, flat: np.ndarray):
        self.W = np.array(flat, dtype=np.float64).reshape(self.action_dim, self.obs_dim)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return self.W @ obs
