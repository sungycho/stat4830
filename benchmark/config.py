"""Experiment configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class EnvConfig:
    name: str = "lqr"  # "lqr" or "pendulum"
    state_dim: int = 4
    action_dim: int = 2
    horizon: int = 200
    noise_std: float = 0.1
    system_seed: int = 42


@dataclass
class MethodConfig:
    name: str = "ars"  # "ars", "vanilla_es", or "reinforce"
    sigma: float = 0.03
    lr: float = 0.02
    N: int = 16
    b: int = 8
    use_state_norm: bool = True
    reward_norm: bool = True
    # REINFORCE-specific
    episodes_per_update: int = 16
    action_sigma: float = 0.5
    baseline: str = "mean"


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    total_episode_budget: int = 3200
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_steps: int = 200
    eval_every_iters: int = 10
    eval_episodes: int = 5
    results_dir: str = "benchmark_results"
    run_tag: str = ""

    def episodes_per_iter(self) -> int:
        name = self.method.name
        if name == "ars":
            return 2 * self.method.N
        elif name == "vanilla_es":
            return self.method.N
        elif name == "reinforce":
            return self.method.episodes_per_update
        raise ValueError(f"Unknown method: {name}")

    def num_iters(self) -> int:
        return self.total_episode_budget // self.episodes_per_iter()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        env = EnvConfig(**d.pop("env"))
        method = MethodConfig(**d.pop("method"))
        return cls(env=env, method=method, **d)

    @classmethod
    def from_json(cls, s: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(s))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_json(Path(path).read_text())


@dataclass
class SweepConfig:
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    variants: list[dict] = field(default_factory=list)
    sweep_name: str = "sweep"

    def apply_overrides(self, overrides: dict) -> ExperimentConfig:
        """Apply dot-path overrides to a copy of the base config."""
        d = self.base.to_dict()
        for key, value in overrides.items():
            parts = key.split(".")
            target = d
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        return ExperimentConfig.from_dict(d)
