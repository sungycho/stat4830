from dataclasses import dataclass
from typing import Dict, Tuple, List

from src.policy_prompt import BernoulliPromptPolicy, render_prompt
from src.utils_prime import prime_eval


@dataclass
class PGConfig:
    lr: float = 0.5
    episodes_per_update: int = 6  # prompt policy를 샘플링하는 횟수
    baseline: str = "mean"  # "mean" or "ema"
    ema_beta: float = 0.9


def run_pg_step(
    policy: BernoulliPromptPolicy,
    cfg: PGConfig,
    env_id: str,
    model: str,
    eval_kwargs: dict,
    baseline_state: Dict[str, float],
) -> Tuple[BernoulliPromptPolicy, Dict, Dict[str, float]]:
    """
    One REINFORCE step on prompt-selection policy:
      a ~ π_theta(a)   (a = which modules active)
      r = reward(prompt(a))
      θ <- θ + lr * (r - b) * ∇ log πθ(a)
    """
    acts: List[Dict[str, bool]] = []
    rewards: List[float] = []

    for _ in range(cfg.episodes_per_update):
        active = policy.sample()
        prompt = render_prompt(active)
        res = prime_eval(env_id=env_id, model=model, prompt=prompt, **eval_kwargs)
        acts.append(active)
        rewards.append(res.reward)

    # compute baseline
    if cfg.baseline == "mean":
        b = sum(rewards) / len(rewards)
    elif cfg.baseline == "ema":
        old = baseline_state.get("ema", 0.0)
        b = old
        baseline_state["ema"] = cfg.ema_beta * old + (1.0 - cfg.ema_beta) * (sum(rewards) / len(rewards))
    else:
        b = 0.0

    # update per-sample (stochastic PG)
    for active, r in zip(acts, rewards):
        policy.update_reinforce(active=active, reward=r, baseline=b, lr=cfg.lr)

    info = {
        "rewards": rewards,
        "baseline": b,
        "logits": dict(policy.logits),
    }
    return policy, info, baseline_state
