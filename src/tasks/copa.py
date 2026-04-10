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

    def build_prompt_base(self, example):
        q = example["question"]
        return (
            'Premise: "The man lost his footing on the ice."\n'
            "What is the effect?\n"
            '1: "He slipped and fell."\n'
            '2: "He ran a marathon."\n'
            "Answer: 1\n\n"
            'Premise: "The woman felt exhausted after work."\n'
            "What is the cause?\n"
            '1: "She had slept for ten hours."\n'
            '2: "She had worked a double shift."\n'
            "Answer: 2\n\n"
            f'Premise: "{example["premise"]}"\n'
            f"What is the {q}?\n"
            f'1: "{example["choice1"]}"\n'
            f'2: "{example["choice2"]}"\n'
            "Answer:"
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
