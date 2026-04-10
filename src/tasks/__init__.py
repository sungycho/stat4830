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

    def build_prompt_base(self, example: dict) -> str:
        """Few-shot completion-style prompt for base models (no chat template).

        Defaults to build_prompt so existing tasks work without any changes.
        Override in subclasses to provide in-context examples that help base
        models infer the expected completion pattern.
        """
        return self.build_prompt(example)

    @abstractmethod
    def score(self, text: str, example: dict) -> float:
        """Score one model generation. Return +1.0 correct, -1.0 wrong/unparseable."""
        ...

    def label_words(self) -> list[str] | None:
        """Return label words for CE scoring (e.g. ['yes', 'no']).

        Return None if CE scoring is not supported for this task (e.g. generation
        tasks or multi-word labels with ambiguous first tokens).
        The order must be consistent with score_ce().
        """
        return None

    def score_ce(self, log_probs: dict[str, float], example: dict) -> float:
        """Score using restricted log-probabilities over label_words().

        log_probs: {word: log P(word | prompt)} restricted log-softmax over
                   label_words(), as returned by backend.score_logprobs_batch().
        Returns the log-prob of the correct label (higher = better).
        Only called when label_words() is not None.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement score_ce")


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
from src.tasks.mnli import MnliTask              # noqa: E402, F401
from src.tasks.cb import CbTask                  # noqa: E402, F401
from src.tasks.wsc import WscTask                # noqa: E402, F401
from src.tasks.wic import WicTask                # noqa: E402, F401
from src.tasks.copa import CopaTask              # noqa: E402, F401
from src.tasks.squad import SquadTask            # noqa: E402, F401
from src.tasks.record import RecordTask          # noqa: E402, F401
from src.tasks.drop import DropTask              # noqa: E402, F401
from src.tasks.gsm8k import Gsm8kTask           # noqa: E402, F401
from src.tasks.math500 import Math500Task        # noqa: E402, F401
from src.tasks.trec import TrecTask              # noqa: E402, F401
from src.tasks.sst5 import Sst5Task              # noqa: E402, F401
