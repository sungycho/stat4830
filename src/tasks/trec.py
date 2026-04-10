from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

# TREC coarse labels (label-coarse field, 0-5)
_LABELS = ["abbreviation", "entity", "description", "human", "location", "numeric"]

# Build a regex per label that matches the first word
_PATTERNS = {label: re.compile(rf"\b{label}\b", re.IGNORECASE) for label in _LABELS}


@register("trec")
class TrecTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("trec")
        train = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
        val_pool = ds["test"]  # TREC has no validation split; test set is small (500)
        val = val_pool.shuffle(seed=seed).select(range(min(val_size, len(val_pool))))
        return _to_list(train, "coarse_label"), _to_list(val, "coarse_label")

    def build_prompt(self, example):
        return (
            f"Classify the question into one of these types: "
            f"abbreviation, entity, description, human, location, numeric.\n"
            f"Question: {example['text']}\n"
            f"Type:"
        )

    def score(self, text, example):
        first_match = None
        first_pos = len(text) + 1
        for label, pat in _PATTERNS.items():
            m = pat.search(text)
            if m and m.start() < first_pos:
                first_match = label
                first_pos = m.start()
        if first_match is None:
            return -1.0
        pred = _LABELS.index(first_match)
        return 1.0 if pred == example["label"] else -1.0


def _to_list(split, label_field):
    return [{"text": ex["text"], "label": ex[label_field]} for ex in split]
