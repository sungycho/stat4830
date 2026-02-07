import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.policy_prompt import ContinuousPromptParams, render_prompt
from src.utils_prime import prime_eval


@dataclass
class ESConfig:
    sigma: float = 0.5
    lr: float = 0.5
    population: int = 6
    reward_baseline: str = "mean"  # "mean" or "none"


def _add_noise(weights: Dict[str, float], eps: Dict[str, float], sigma: float) -> Dict[str, float]:
    return {k: weights[k] + sigma * eps[k] for k in weights}


def _sample_eps(keys: List[str]) -> Dict[str, float]:
    return {k: random.gauss(0.0, 1.0) for k in keys}


def run_es_step(
    theta: ContinuousPromptParams,
    cfg: ESConfig,
    env_id: str,
    model: str,
    eval_kwargs: dict,
) -> Tuple[ContinuousPromptParams, Dict]:
    """
    One ES update step:
      eps_i ~ N(0, I)
      r_i = reward(theta + sigma*eps_i)
      theta <- theta + lr * (1/N) * sum_i ( (r_i - b) * eps_i )
    """
    keys = list(theta.weights.keys())

    eps_list: List[Dict[str, float]] = []
    rewards: List[float] = []

    # rollout population
    for _ in range(cfg.population):
        eps = _sample_eps(keys)
        w_pert = _add_noise(theta.weights, eps, cfg.sigma)
        active = {k: (w_pert[k] > 0.0) for k in keys}
        prompt = render_prompt(active)

        res = prime_eval(env_id=env_id, model=model, prompt=prompt, **eval_kwargs)
        eps_list.append(eps)
        rewards.append(res.reward)

    # baseline
    b = sum(rewards) / len(rewards) if (cfg.reward_baseline == "mean") else 0.0

    # gradient estimate
    grad = {k: 0.0 for k in keys}
    for eps, r in zip(eps_list, rewards):
        adv = r - b
        for k in keys:
            grad[k] += adv * eps[k]
    for k in keys:
        grad[k] /= float(cfg.population)

    # update
    new_weights = {k: theta.weights[k] + cfg.lr * grad[k] for k in keys}
    info = {
        "rewards": rewards,
        "baseline": b,
        "grad": grad,
        "new_weights": new_weights,
    }
    return ContinuousPromptParams(new_weights), info
