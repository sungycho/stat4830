"""MuJoCo environment wrappers matching the paper's evaluation tasks.

Supports the 6 locomotion tasks from Mania et al. NeurIPS 2018:
  swimmer, hopper, halfcheetah, walker2d, ant, humanoid

Tasks with a survival bonus (hopper, walker2d, ant, humanoid) can have
that +1/step bonus stripped during training via remove_survival_bonus=True.
Pass remove_survival_bonus=False for eval to use the standard reward.

Requires: gymnasium[mujoco] or mujoco-py + gym.
"""

from __future__ import annotations

import numpy as np

# Map task name → gymnasium env id (prefer v5; fall back to v4 if unavailable)
_TASK_TO_GYM_ID = {
    "swimmer": "Swimmer-v5",
    "hopper": "Hopper-v5",
    "halfcheetah": "HalfCheetah-v5",
    "walker2d": "Walker2d-v5",
    "ant": "Ant-v5",
    "humanoid": "Humanoid-v5",
}

_TASK_TO_GYM_ID_FALLBACK = {
    "swimmer": "Swimmer-v4",
    "hopper": "Hopper-v4",
    "halfcheetah": "HalfCheetah-v4",
    "walker2d": "Walker2d-v4",
    "ant": "Ant-v4",
    "humanoid": "Humanoid-v4",
}

# Tasks that include a +1/step survival bonus in their standard reward
_SURVIVAL_BONUS_TASKS = {"hopper", "walker2d", "ant", "humanoid"}


class MuJoCoEnv:
    """Thin wrapper around a gymnasium MuJoCo task.

    Provides the same interface as LQREnv/PendulumEnv:
      reset(seed) → obs
      step(action) → (obs, reward, done)
      close()

    Parameters
    ----------
    task_name:
        One of swimmer, hopper, halfcheetah, walker2d, ant, humanoid.
    remove_survival_bonus:
        If True and the task has a survival bonus, subtract 1.0 from every
        per-step reward during training.  Set False for eval environments.
    max_steps:
        Override the environment's default episode horizon if > 0.
    """

    def __init__(
        self,
        task_name: str,
        remove_survival_bonus: bool = False,
        max_steps: int = 0,
    ):
        import gymnasium as gym

        task_name = task_name.lower()
        if task_name not in _TASK_TO_GYM_ID:
            raise ValueError(
                f"Unknown MuJoCo task: {task_name!r}. "
                f"Valid tasks: {sorted(_TASK_TO_GYM_ID)}"
            )

        gym_id = _TASK_TO_GYM_ID[task_name]
        try:
            self._env = gym.make(gym_id)
        except Exception:
            # Try the v4 fallback (some installations may not have v5)
            gym_id = _TASK_TO_GYM_ID_FALLBACK[task_name]
            try:
                self._env = gym.make(gym_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create gymnasium env for task '{task_name}'. "
                    "Ensure gymnasium[mujoco] and mujoco are installed: "
                    "pip install gymnasium[mujoco] mujoco"
                ) from exc

        self.task_name = task_name
        self._remove_survival_bonus = (
            remove_survival_bonus and task_name in _SURVIVAL_BONUS_TASKS
        )

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        self.obs_dim: int = int(np.prod(obs_space.shape))
        self.action_dim: int = int(np.prod(act_space.shape))
        self.action_low: np.ndarray = act_space.low.astype(np.float64)
        self.action_high: np.ndarray = act_space.high.astype(np.float64)

        self._max_steps = max_steps if max_steps > 0 else None
        self._t = 0

    def reset(self, seed: int = 0) -> np.ndarray:
        obs, _ = self._env.reset(seed=int(seed))
        self._t = 0
        return obs.astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        action = np.asarray(action, dtype=np.float64).ravel()
        obs, reward, terminated, truncated, _ = self._env.step(action)
        self._t += 1

        reward = float(reward)
        if self._remove_survival_bonus:
            reward -= 1.0

        done = bool(terminated or truncated)
        if self._max_steps is not None and self._t >= self._max_steps:
            done = True

        return obs.astype(np.float64), reward, done

    def close(self):
        self._env.close()
