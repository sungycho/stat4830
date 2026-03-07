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
            return -1.0
        return 1.0 if pred == example["label"] else -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
