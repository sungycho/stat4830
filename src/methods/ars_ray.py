"""Ray-parallel ARS matching the paper's distributed implementation.

Paper §3 implementation details (Mania et al. NeurIPS 2018):
  - A shared noise table of Gaussian entries is pre-generated and placed in
    Ray's object store once.  Workers identify perturbation directions by a
    (table, index) pair rather than receiving full noise vectors, so the only
    per-iteration communication is small index arrays and scalar rewards.
  - N Ray tasks are dispatched concurrently per iteration; each task runs the
    positive then negative rollout for one direction sequentially.
  - Wall-clock time per iteration ≈ cost of two episodes (the sequential cost
    within one task), not one — contrast with a fully pipelined implementation
    where pos and neg rollouts were also parallelised.

Running-norm update in the parallel setting:
  Paper does not specify how the shared RunningNorm is updated in the
  distributed case.  We use Chan's parallel Welford formula:
    1. At the start of iteration t, master snapshots norm state S_t.
    2. Workers normalise observations with the frozen S_t (no communication).
    3. Workers track their *new* observations in a local accumulator.
    4. Master merges all N accumulators with S_t via Chan's formula → S_{t+1}.
  This is mathematically equivalent to processing all observations in one
  batch, which is the standard approximation in distributed RL.

Gap vs. paper:
  - Paper distributes across multiple machines; this runs on a single node.
  - Each Ray task runs pos + neg rollouts sequentially (2× episode cost per
    task batch, not 1×); the paper's workers each handle one rollout.
  - Noise table size is configurable via MethodConfig.noise_table_size or the
    ARS_NOISE_TABLE_SIZE env var (paper uses 250 M entries ≈ 1 GB).
  - State-norm update is a parallel-batch approximation (see above).
"""

from __future__ import annotations

import os
import numpy as np
import ray

from src.envs.base import rollout, eval_policy
from src.methods.ars import _build_eval_env_if_needed
from src.methods.base import MethodResult
from src.policy import LinearPolicy, RunningNorm


# ---------------------------------------------------------------------------
# Noise table
# ---------------------------------------------------------------------------

_NOISE_TABLE_SIZE = 250_000_000   # paper default; ~1 GB float32
_NOISE_TABLE_SEED = 12345


def _noise_table_cap(explicit: int = 0) -> int:
    """Return the effective noise table size cap.

    Priority (highest first):
      1. explicit argument (MethodConfig.noise_table_size > 0)
      2. ARS_NOISE_TABLE_SIZE environment variable
      3. _NOISE_TABLE_SIZE module default (250 M)
    """
    if explicit > 0:
        return explicit
    env_val = os.environ.get("ARS_NOISE_TABLE_SIZE", "")
    if env_val.strip():
        try:
            return int(env_val)
        except ValueError:
            pass
    return _NOISE_TABLE_SIZE


def _build_noise_table(theta_dim: int, N: int, num_iters: int, max_size: int = 0) -> np.ndarray:
    """Pre-generate Gaussian noise table (paper §3: shared noise table).

    Size is the smaller of `max_size` (or the module default 250 M when 0)
    and a 4× safety margin over the maximum indices needed.  Override via
    MethodConfig.noise_table_size or the ARS_NOISE_TABLE_SIZE env var.
    """
    cap = _noise_table_cap(max_size)
    needed = max(theta_dim * N * num_iters * 4, 1_000_000)
    size = min(needed, cap)
    rng = np.random.RandomState(_NOISE_TABLE_SEED)
    return rng.standard_normal(size).astype(np.float32)


def _get_delta(table: np.ndarray, idx: int, dim: int) -> np.ndarray:
    """Extract `dim` entries from noise table starting at `idx`, wrapping."""
    end = idx + dim
    if end <= len(table):
        return table[idx:end].astype(np.float64)
    return np.concatenate(
        [table[idx:], table[: end - len(table)]]
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Running norm: parallel Welford merge (Chan's formula)
# ---------------------------------------------------------------------------

def _merge_norm_states(base: dict, *extras: dict) -> dict:
    """Combine multiple RunningNorm states via Chan's parallel Welford formula.

    `base`  : current master state S_t (includes all history so far).
    `extras`: per-worker fresh states containing ONLY new observations
              (not including the historical count/mean/M2 from S_t).

    Returns S_{t+1} = S_t ∪ all observations seen by all workers.
    """
    count = base["count"]
    mean  = np.array(base["mean"], dtype=np.float64)
    M2    = np.array(base["M2"],   dtype=np.float64)

    for extra in extras:
        n_b = extra["count"]
        if n_b == 0:
            continue
        n_a    = count
        n      = n_a + n_b
        mean_b = np.array(extra["mean"], dtype=np.float64)
        M2_b   = np.array(extra["M2"],   dtype=np.float64)
        d      = mean_b - mean
        mean   = (n_a * mean + n_b * mean_b) / n
        M2     = M2 + M2_b + d ** 2 * (n_a * n_b) / n
        count  = n

    return {
        "dim":   base["dim"],
        "eps":   base["eps"],
        "count": count,
        "mean":  mean.tolist(),
        "M2":    M2.tolist(),
    }


class _WorkerNorm:
    """Running-norm proxy for use inside Ray workers.

    Normalises observations using a frozen snapshot of the master norm S_t
    (no update, so the normalisation is consistent across all parallel workers).
    Simultaneously accumulates new observations in a fresh local tracker so
    the master can merge them back using Chan's formula after the iteration.
    """

    def __init__(self, frozen_state: dict | None):
        if frozen_state is not None:
            self._frozen  = RunningNorm.from_state_dict(frozen_state)
            self._tracker = RunningNorm(frozen_state["dim"], frozen_state["eps"])
        else:
            self._frozen  = None
            self._tracker = None

    def update(self, x: np.ndarray) -> None:
        if self._tracker is not None:
            self._tracker.update(x)          # track new obs only

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self._frozen is not None:
            return self._frozen.normalize(x)  # use frozen snapshot
        return np.asarray(x, dtype=np.float64)

    def tracker_state(self) -> dict | None:
        return self._tracker.state_dict() if self._tracker is not None else None


# ---------------------------------------------------------------------------
# Environment builder for workers
# ---------------------------------------------------------------------------

def _build_env_in_worker(env_name: str, env_kwargs: dict):
    """Reconstruct an environment inside a Ray worker process."""
    if env_name == "lqr":
        from src.envs.lqr import LQREnv
        return LQREnv(**env_kwargs)
    elif env_name == "pendulum":
        from src.envs.pendulum import PendulumEnv
        return PendulumEnv(**env_kwargs)
    else:
        from src.envs.mujoco import MuJoCoEnv
        return MuJoCoEnv(**env_kwargs)


def _env_name_and_kwargs(config) -> tuple[str, dict]:
    """Extract env name and serialisable constructor kwargs from ExperimentConfig."""
    ec = config.env
    if ec.name == "lqr":
        return "lqr", {
            "state_dim":   ec.state_dim,
            "action_dim":  ec.action_dim,
            "horizon":     ec.horizon,
            "noise_std":   ec.noise_std,
            "system_seed": ec.system_seed,
        }
    elif ec.name == "pendulum":
        return "pendulum", {
            "obs_noise_std": ec.noise_std,
            "max_steps":     config.max_steps,
        }
    else:
        return ec.name, {
            "task_name":             ec.name,
            "remove_survival_bonus": ec.remove_survival_bonus,
            "max_steps":             config.max_steps,
        }


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------

@ray.remote
def _eval_direction(
    theta_flat:     np.ndarray,
    noise_table_ref,              # Ray ObjectRef → shared noise table
    noise_idx:      int,
    theta_dim:      int,
    obs_dim:        int,
    action_dim:     int,
    sigma:          float,
    env_name:       str,
    env_kwargs:     dict,
    max_steps:      int,
    seed_pos:       int,
    seed_neg:       int,
    norm_state:     dict | None,  # frozen master norm used for normalisation
) -> tuple[float, float, dict | None]:
    """Evaluate one perturbation direction (+/-) in a Ray worker.

    Returns
    -------
    (r_pos, r_neg, fresh_obs_state)
      r_pos / r_neg     : episode returns for positive / negative perturbation.
      fresh_obs_state   : RunningNorm state dict of ONLY the observations seen
                          during this worker's two rollouts (count starts at 0).
                          None when state normalisation is disabled.
    """
    # ── Retrieve delta from shared noise table ────────────────────────────────
    # Ray automatically dereferences ObjectRef args, so noise_table_ref is
    # already the ndarray here (no ray.get() needed inside the worker).
    delta = _get_delta(noise_table_ref, noise_idx, theta_dim)

    # ── Reconstruct environment ───────────────────────────────────────────────
    env = _build_env_in_worker(env_name, env_kwargs)

    # ── Positive perturbation ─────────────────────────────────────────────────
    norm_pos = _WorkerNorm(norm_state)
    theta_pos = theta_flat + sigma * delta
    W_pos = theta_pos.reshape(action_dim, obs_dim)
    r_pos, _ = rollout(env, lambda obs, W=W_pos: W @ obs, seed_pos, max_steps, norm_pos)

    # ── Negative perturbation (fresh tracker, same frozen snapshot) ───────────
    norm_neg = _WorkerNorm(norm_state)
    theta_neg = theta_flat - sigma * delta
    W_neg = theta_neg.reshape(action_dim, obs_dim)
    r_neg, _ = rollout(env, lambda obs, W=W_neg: W @ obs, seed_neg, max_steps, norm_neg)

    env.close()

    # ── Merge pos + neg trackers → single fresh state for the master ──────────
    fresh_state: dict | None = None
    if norm_state is not None:
        ts_pos = norm_pos.tracker_state()
        ts_neg = norm_neg.tracker_state()
        empty  = {
            "dim":   norm_state["dim"],
            "eps":   norm_state["eps"],
            "count": 0,
            "mean":  [0.0] * norm_state["dim"],
            "M2":    [0.0] * norm_state["dim"],
        }
        fresh_state = _merge_norm_states(empty, ts_pos, ts_neg)

    return float(r_pos), float(r_neg), fresh_state


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_ars_ray(config, env, seed: int) -> list[MethodResult]:
    """ARS training with Ray-parallel rollout collection.

    Drop-in replacement for ``run_ars``; enable via ``MethodConfig.use_ray=True``.

    The function initialises Ray if it is not already running.  Call
    ``ray.init(num_cpus=N)`` before this function to control parallelism.

    Key differences from sequential ``run_ars``:
      - N direction pairs per iteration run as concurrent Ray tasks.
      - Perturbation noise is drawn from a shared noise table in Ray's
        object store, never re-transmitted (paper §3 design).
      - RunningNorm is updated via Chan's parallel-Welford merge, not
        by sequential in-rollout updates.
    """
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Ray init failed ({exc}); falling back to sequential run_ars.",
                stacklevel=2,
            )
            from src.methods.ars import run_ars
            return run_ars(config, env, seed)

    mc        = config.method
    num_iters = config.num_iters()

    policy       = LinearPolicy(env.obs_dim, env.action_dim)
    running_norm = RunningNorm(env.obs_dim) if mc.use_state_norm else None
    theta_dim    = policy.theta.size

    eval_env      = _build_eval_env_if_needed(config, env)
    owns_eval_env = eval_env is not env

    # ── Pre-generate noise table; put in Ray object store once ───────────────
    noise_table     = _build_noise_table(theta_dim, mc.N, num_iters, mc.noise_table_size)
    noise_table_ref = ray.put(noise_table)
    max_start_idx   = max(len(noise_table) - theta_dim, 1)

    env_name, env_kwargs = _env_name_and_kwargs(config)

    rng               = np.random.default_rng(seed)
    episodes_consumed = 0
    results           = []

    for t in range(num_iters):
        # Sample one noise-table index per direction (reproducible via seed)
        noise_indices = [int(rng.integers(0, max_start_idx)) for _ in range(mc.N)]

        theta_flat = policy.theta.copy()
        norm_state = running_norm.state_dict() if running_norm is not None else None

        # ── Dispatch N tasks in parallel ──────────────────────────────────────
        futures = [
            _eval_direction.remote(
                theta_flat,
                noise_table_ref,
                noise_indices[k],
                theta_dim,
                env.obs_dim,
                env.action_dim,
                mc.sigma,
                env_name,
                env_kwargs,
                config.max_steps,
                seed * 1_000_000 + t * 1000 + 2 * k,       # seed_pos (matches sequential)
                seed * 1_000_000 + t * 1000 + 2 * k + 1,   # seed_neg
                norm_state,
            )
            for k in range(mc.N)
        ]
        worker_out = ray.get(futures)

        rewards_pos  = [w[0] for w in worker_out]
        rewards_neg  = [w[1] for w in worker_out]
        fresh_states = [w[2] for w in worker_out]
        episodes_consumed += 2 * mc.N

        # ── Update running norm via Chan's parallel Welford merge ─────────────
        if running_norm is not None and norm_state is not None:
            valid = [s for s in fresh_states if s is not None]
            if valid:
                merged = _merge_norm_states(norm_state, *valid)
                running_norm = RunningNorm.from_state_dict(merged)

        # ── Reconstruct deltas locally (no communication; same noise table) ───
        deltas = [
            _get_delta(noise_table, noise_indices[k], theta_dim)
            for k in range(mc.N)
        ]

        # ── Top-b selection ───────────────────────────────────────────────────
        max_rewards = [max(rp, rn) for rp, rn in zip(rewards_pos, rewards_neg)]
        top_idx     = np.argsort(max_rewards)[-mc.b:]

        # ── Policy update (same formula as sequential run_ars) ────────────────
        step = np.zeros_like(policy.theta)
        for i in top_idx:
            step += (rewards_pos[i] - rewards_neg[i]) * deltas[i]

        if mc.reward_norm:
            rewards_used = [rewards_pos[i] for i in top_idx] + [rewards_neg[i] for i in top_idx]
            sigma_R      = max(np.std(rewards_used), 1e-8)
            policy.theta = policy.theta + (mc.lr / (mc.b * sigma_R)) * step
        else:
            policy.theta = policy.theta + (mc.lr / mc.b) * step

        # ── Eval ──────────────────────────────────────────────────────────────
        eval_ret = None
        if (t + 1) % config.eval_every_iters == 0 or t == num_iters - 1:
            eval_seed = seed * 1_000_000 + 999_000 + t
            W_eval    = policy.theta.reshape(env.action_dim, env.obs_dim)
            eval_ret  = eval_policy(
                eval_env,
                lambda obs, W=W_eval: W @ obs,
                eval_seed,
                config.eval_episodes,
                config.max_steps,
                running_norm,
            )

        results.append(MethodResult(
            iteration=t,
            episodes_consumed=episodes_consumed,
            train_returns=rewards_pos + rewards_neg,
            eval_return=eval_ret,
        ))

    if owns_eval_env:
        eval_env.close()

    return results
