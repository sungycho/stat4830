from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_POS = re.compile(r"\bpositive\b", re.IGNORECASE)
_NEG = re.compile(r"\bnegative\b", re.IGNORECASE)
_LABEL_MAP = {0: "negative", 1: "positive"}


@register("sst2")
class Sst2Task(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("glue", "sst2")
        train = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(range(min(val_size, len(val_pool))))
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f'Classify the sentiment of this sentence.\n'
            f'Sentence: "{example["sentence"]}"\n'
            f"Answer with exactly one word — positive or negative:"
        )

    def score(self, text, example):
        pos = _POS.search(text)
        neg = _NEG.search(text)
        if pos and neg:
            pred = "positive" if pos.start() < neg.start() else "negative"
        elif pos:
            pred = "positive"
        elif neg:
            pred = "negative"
        else:
            return -1.0
        return 1.0 if pred == _LABEL_MAP[example["label"]] else -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
