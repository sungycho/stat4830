from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]

# Match multi-word labels before single-word ones to avoid partial hits
_PATTERNS = [
    (0, re.compile(r"\bvery\s+negative\b", re.IGNORECASE)),
    (4, re.compile(r"\bvery\s+positive\b", re.IGNORECASE)),
    (1, re.compile(r"\bnegative\b",        re.IGNORECASE)),
    (3, re.compile(r"\bpositive\b",        re.IGNORECASE)),
    (2, re.compile(r"\bneutral\b",         re.IGNORECASE)),
]


@register("sst5")
class Sst5Task(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("sst", "default")
        train = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(range(min(val_size, len(val_pool))))
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f"Classify the sentiment of the sentence. "
            f"Choose one: very negative, negative, neutral, positive, very positive.\n"
            f'Sentence: "{example["sentence"]}"\n'
            f"Sentiment:"
        )

    def score(self, text, example):
        first_label = None
        first_pos = len(text) + 1
        for label_id, pat in _PATTERNS:
            m = pat.search(text)
            if m and m.start() < first_pos:
                first_label = label_id
                first_pos = m.start()
        if first_label is None:
            return -1.0
        return 1.0 if first_label == example["label"] else -1.0


def _to_list(split):
    # SST fine-grained: label is a float in [0,1]; bin into 5 classes
    results = []
    for ex in split:
        score = float(ex["label"])
        if score <= 0.2:
            label = 0
        elif score <= 0.4:
            label = 1
        elif score <= 0.6:
            label = 2
        elif score <= 0.8:
            label = 3
        else:
            label = 4
        results.append({"sentence": ex["sentence"], "label": label})
    return results
