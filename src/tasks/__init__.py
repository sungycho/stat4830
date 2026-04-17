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

    @property
    def prefer_base_prompt(self) -> bool:
        """If True, always use build_prompt_base() regardless of backend.is_instruct.

        Override to True in tasks that pre-format their own prompts (e.g. countdown),
        where applying a chat template on top would break the prompt structure.
        """
        return False

    def predict(self, text: str) -> str | None:
        """Extract the predicted label string from model output. None = parse failure.

        Used for prediction distribution analysis (not scoring/training).
        Override in each task alongside score().
        """
        return None

    def gold_label(self, example: dict) -> str:
        """Return the gold label string for an example. Used for distribution analysis."""
        return str(example.get("label", "?"))

    def reward(self, text: str, example: dict) -> float:
        """Training reward signal. Defaults to remapping score() {-1,+1} → {0,1}.

        Override in tasks that need a richer reward (e.g. partial credit for format).
        """
        return 1.0 if self.score(text, example) > 0 else 0.0

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

    def build_prompt_mezo(self, example: dict) -> str:
        """MeZO-style completion prompt (paper 2305.17333, Appendix Table 14).

        Defaults to build_prompt_base so tasks without an override fall back gracefully.
        Override in subclasses to use the exact template from the MeZO paper.
        """
        return self.build_prompt_base(example)

    def label_words_mezo(self) -> list[str] | None:
        """Label words for CE scoring under the MeZO prompt style.

        Some tasks use different label words in MeZO prompts (e.g. SST-2 uses
        'great'/'terrible' instead of 'positive'/'negative'; CB uses 'Yes'/'No'/'Maybe').
        Defaults to label_words() for tasks where the labels are unchanged.
        """
        return self.label_words()

    def score_mezo(self, text: str, example: dict) -> float:
        """Score generated text when using MeZO-style prompts.

        Defaults to score() for tasks where labels are unchanged.
        Override in tasks where MeZO prompts elicit different label words
        (e.g. SST-2 expects 'great'/'terrible', CB expects 'Yes'/'No'/'Maybe').
        """
        return self.score(text, example)

    def score_ce_mezo(self, log_probs: dict[str, float], example: dict) -> float:
        """CE score using restricted log-probs under the MeZO prompt style.

        Returns the log-prob of the correct label word from label_words_mezo().
        Defaults to score_ce() for tasks where MeZO label words are unchanged.
        Override in tasks that use different label words (SST-2, CB already do).
        """
        return self.score_ce(log_probs, example)


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
