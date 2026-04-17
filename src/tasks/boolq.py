from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)

# BoolQ labels: 0 = False (no), 1 = True (yes)
# Passages can be very long — truncate to keep prompts manageable at max_new_tokens=4.
_PASSAGE_MAX_CHARS = 400


@register("boolq")
class BoolqTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "boolq")
        train = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(range(min(val_size, len(val_pool))))
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            f'Passage: "{passage}"\n'
            f'Question: {example["question"]}?\n'
            f"Answer yes or no:"
        )

    def build_prompt_base(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            'Passage: "Mercury is the smallest planet in the solar system and the closest to the Sun."\n'
            "Question: is Mercury the largest planet?\n"
            "Answer: no\n\n"
            'Passage: "Water boils at 100 degrees Celsius at standard atmospheric pressure."\n'
            "Question: does water boil at 100 degrees Celsius?\n"
            "Answer: yes\n\n"
            f'Passage: "{passage}"\n'
            f"Question: {example['question']}?\n"
            "Answer:"
        )

    def build_prompt_free(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        return f'{passage}\n{example["question"]}?'

    def predict(self, text: str) -> str | None:
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            return "yes" if yes.start() < no.start() else "no"
        return "yes" if yes else ("no" if no else None)

    def gold_label(self, example: dict) -> str:
        return "yes" if example["label"] == 1 else "no"

    def label_words(self):
        return ["yes", "no"]

    def score_ce(self, log_probs, example):
        correct = "yes" if example["label"] == 1 else "no"
        return log_probs[correct]

    def score(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 1 if yes.start() < no.start() else 0
        elif yes:
            pred = 1  # True
        elif no:
            pred = 0  # False
        else:
            return 0.0  # parse failure
        return 1.0 if pred == example["label"] else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): "<passage> <question>?" — no structural labels
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return f'{passage} {example["question"]}?'

    def label_words_mezo(self):
        return ["Yes", "No"]

    def score_mezo(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 1 if yes.start() < no.start() else 0
        elif yes:
            pred = 1  # True
        elif no:
            pred = 0  # False
        else:
            return -1.0
        return 1.0 if pred == example["label"] else -1.0

    def score_ce_mezo(self, log_probs, example):
        correct = "Yes" if example["label"] == 1 else "No"
        return log_probs[correct]


def _to_list(split):
    return [dict(ex) for ex in split]
