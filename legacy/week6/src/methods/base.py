"""Result contract for all methods."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MethodResult:
    iteration: int
    episodes_consumed: int
    train_returns: list[float] = field(default_factory=list)
    eval_return: float | None = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "episodes_consumed": self.episodes_consumed,
            "train_returns": self.train_returns,
            "eval_return": self.eval_return,
        }
