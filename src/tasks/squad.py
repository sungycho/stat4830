from __future__ import annotations
import re
import string
from datasets import load_dataset
from src.tasks import Task, register

_CONTEXT_MAX_CHARS = 600


def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return " ".join(text.split())


@register("squad")
class SquadTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("squad")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        ctx = example["context"][:_CONTEXT_MAX_CHARS]
        if len(example["context"]) > _CONTEXT_MAX_CHARS:
            ctx += "..."
        return (
            f"Context: {ctx}\n"
            f'Question: {example["question"]}\n'
            f"Answer in as few words as possible:"
        )

    def score(self, text, example):
        pred = _normalize(text)
        if not pred:
            return -1.0
        gold_answers = example["answers"]["text"]
        for ans in gold_answers:
            if _normalize(ans) in pred:
                return 1.0
        return -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
