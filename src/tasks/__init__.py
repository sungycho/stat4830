"""Task registry for ES fine-tuning.

To add a new task:
1. Create src/tasks/<name>.py
2. Subclass Task, implement load_data / build_prompt / score
3. Decorate the class with @register("<name>")
4. Add its import at the bottom of this file — that's it.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

_REGISTRY: dict[str, type["Task"]] = {}


class Task(ABC):
    """Common interface every classification task must implement."""

    @abstractmethod
    def load_data(
        self, train_size: int, val_size: int, seed: int
    ) -> tuple[list[dict], list[dict]]:
        """Return (train_examples, val_examples) as plain lists of dicts."""
        ...

    @abstractmethod
    def build_prompt(self, example: dict) -> str:
        """Build the prompt string for a single example."""
        ...

    @abstractmethod
    def score(self, text: str, example: dict) -> float:
        """Score one model generation. Return +1.0 correct, -1.0 wrong/unparseable."""
        ...


def register(name: str):
    """Class decorator that adds a Task subclass to the registry."""
    def decorator(cls: type[Task]) -> type[Task]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {available_tasks()}")
    return _REGISTRY[name]()


def available_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


# Register all tasks by importing them here.
# Adding a new task only requires creating the file and adding one line below.
from src.tasks.sst2 import Sst2Task              # noqa: E402, F401
from src.tasks.rte import RteTask                # noqa: E402, F401
from src.tasks.boolq import BoolqTask            # noqa: E402, F401
from src.tasks.countdown import CountdownTask    # noqa: E402, F401
