from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_POS = re.compile(r"\bpositive\b", re.IGNORECASE)
_NEG = re.compile(r"\bnegative\b", re.IGNORECASE)
_LABEL_MAP = {0: "negative", 1: "positive"}

_GREAT = re.compile(r"\bgreat\b", re.IGNORECASE)
_TERRIBLE = re.compile(r"\bterrible\b", re.IGNORECASE)
# MeZO: label 1=positive→"great", label 0=negative→"terrible"
_MEZO_LABEL_MAP = {0: "terrible", 1: "great"}


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

    def build_prompt_base(self, example):
        return (
            'Sentence: "A masterpiece of cinema — thrilling, emotional, and unforgettable."\n'
            "Sentiment: positive\n\n"
            'Sentence: "Painfully slow and utterly devoid of any interesting ideas."\n'
            "Sentiment: negative\n\n"
            f'Sentence: "{example["sentence"]}"\n'
            "Sentiment:"
        )

    def label_words(self):
        return ["positive", "negative"]

    def score_ce(self, log_probs, example):
        correct = _LABEL_MAP[example["label"]]
        return log_probs[correct]

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


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): "<text> It was" → model completes with "great"/"terrible"
        return f'{example["sentence"]} It was'

    def label_words_mezo(self):
        return ["great", "terrible"]

    def score_mezo(self, text, example):
        great = _GREAT.search(text)
        terrible = _TERRIBLE.search(text)
        if great and terrible:
            pred = 1 if great.start() < terrible.start() else 0
        elif great:
            pred = 1
        elif terrible:
            pred = 0
        else:
            return -1.0
        return 1.0 if pred == example["label"] else -1.0

    def score_ce_mezo(self, log_probs, example):
        correct = _MEZO_LABEL_MAP[example["label"]]
        return log_probs[correct]


def _to_list(split):
    return [dict(ex) for ex in split]
