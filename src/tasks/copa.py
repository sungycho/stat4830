from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_ONE = re.compile(r"\b1\b")
_TWO = re.compile(r"\b2\b")

# COPA labels: 0 = choice1, 1 = choice2


@register("copa")
class CopaTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "copa")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        q = example["question"]  # "cause" or "effect"
        return (
            f'Premise: "{example["premise"]}"\n'
            f"What is the {q}?\n"
            f'1: "{example["choice1"]}"\n'
            f'2: "{example["choice2"]}"\n'
            f"Answer 1 or 2:"
        )

    def score(self, text, example):
        one = _ONE.search(text)
        two = _TWO.search(text)
        if one and two:
            pred = 0 if one.start() < two.start() else 1
        elif one:
            pred = 0  # choice1
        elif two:
            pred = 1  # choice2
        else:
            return -1.0
        return 1.0 if pred == example["label"] else -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
