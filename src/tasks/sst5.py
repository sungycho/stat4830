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
        ds = load_dataset("SetFit/sst5")
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

    def predict(self, text: str) -> str | None:
        first_label = None
        first_pos = len(text) + 1
        for label_id, pat in _PATTERNS:
            m = pat.search(text)
            if m and m.start() < first_pos:
                first_label = _LABELS[label_id]
                first_pos = m.start()
        return first_label

    def gold_label(self, example: dict) -> str:
        return _LABELS[example["label"]]

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
    # SetFit/sst5: label is already an integer 0-4; field is "text"
    return [{"sentence": ex["text"], "label": int(ex["label"])} for ex in split]
