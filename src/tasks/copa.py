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

    def predict(self, text: str) -> str | None:
        one = _ONE.search(text)
        two = _TWO.search(text)
        if one and two:
            return "1" if one.start() < two.start() else "2"
        return "1" if one else ("2" if two else None)

    def gold_label(self, example: dict) -> str:
        return "1" if example["label"] == 0 else "2"

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
            return 0.0  # parse failure
        return 1.0 if pred == example["label"] else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): "<premise> so/because <candidate>" with per-candidate
        # log-likelihood scoring. Here we approximate with a generation prompt since
        # our framework doesn't yet support full-sentence log-likelihood comparison.
        connector = "so" if example["question"] == "effect" else "because"
        premise = example["premise"].rstrip(".")
        return (
            f'{premise} {connector} {example["choice1"]}, or '
            f'{premise} {connector} {example["choice2"]}? '
            f'Which is more likely: 1 or 2?'
        )

    def label_words_mezo(self):
        return None  # COPA is multiple choice; no single label token for CE scoring

    def score_mezo(self, text, example):
        # Same as score() — looks for "1" or "2" in generated output
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
