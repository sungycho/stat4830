from dataclasses import dataclass
from typing import Dict, List
import math
import random


# Prompt building blocks that can be toggled on/off.
# These represent high-level behavioral rules for the agent.
PROMPT_MODULES: Dict[str, str] = {
    "format": (
        "Output ONLY a single 5-letter guess wrapped as XML:\n"
        "<guess>abcde</guess>\n"
        "No extra text."
    ),
    "track": (
        "Track eliminated letters and confirmed positions. Never repeat eliminated letters."
    ),
    "info_gain": (
        "In early turns, prefer guesses that maximize information gain (cover frequent letters)."
    ),
    "deduce": (
        "Use feedback strictly: greens fixed, yellows included elsewhere, grays excluded."
    ),
}

BASE_SYSTEM = "You are a Wordle-solving agent."


def render_prompt(active: Dict[str, bool]) -> str:
    """
    Construct a system prompt from the base instruction
    and a set of active prompt modules.
    """
    parts: List[str] = [BASE_SYSTEM]
    for k, v in active.items():
        if v and k in PROMPT_MODULES:
            parts.append(PROMPT_MODULES[k])
    return "\n\n".join(parts)


@dataclass
class BernoulliPromptPolicy:
    """
    Policy Gradient baseline.

    Each prompt module is independently included with
    probability sigmoid(logit). The logits are the policy parameters.
    """
    logits: Dict[str, float]

    def sample(self) -> Dict[str, bool]:
        """
        Sample a binary activation decision for each module.
        """
        active = {}
        for k, logit in self.logits.items():
            p = 1.0 / (1.0 + math.exp(-logit))
            active[k] = (random.random() < p)
        return active

    def log_prob(self, active: Dict[str, bool]) -> float:
        """
        Compute log-probability of a sampled activation pattern.
        """
        lp = 0.0
        for k, a in active.items():
            logit = self.logits[k]
            p = 1.0 / (1.0 + math.exp(-logit))
            lp += math.log(p + 1e-12) if a else math.log(1.0 - p + 1e-12)
        return lp

    def update_reinforce(self, active: Dict[str, bool], reward: float, baseline: float, lr: float):
        """
        Perform a REINFORCE update on the logits.

        For Bernoulli policies with logit parameterization:
        grad log pi(a) = a - p
        """
        adv = reward - baseline
        for k, a in active.items():
            logit = self.logits[k]
            p = 1.0 / (1.0 + math.exp(-logit))
            grad = (1.0 if a else 0.0) - p
            self.logits[k] = logit + lr * adv * grad


@dataclass
class ContinuousPromptParams:
    """
    Continuous parameterization used for Evolution Strategies.

    Each module has a real-valued weight.
    A simple threshold determines whether the module is active.
    """
    weights: Dict[str, float]

    def to_active(self, threshold: float = 0.0) -> Dict[str, bool]:
        """
        Convert continuous weights into binary module activations.
        """
        return {k: (w > threshold) for k, w in self.weights.items()}
