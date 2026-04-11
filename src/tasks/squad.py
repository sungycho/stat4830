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

    def build_prompt_base(self, example):
        ctx = example["context"][:_CONTEXT_MAX_CHARS]
        if len(example["context"]) > _CONTEXT_MAX_CHARS:
            ctx += "..."
        return (
            "Context: The Amazon River is the largest river in the world by discharge volume and drainage basin.\n"
            "Question: What is the Amazon River the largest of?\n"
            "Answer: river in the world\n\n"
            "Context: Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.\n"
            "Question: What field did Marie Curie conduct pioneering research in?\n"
            "Answer: radioactivity\n\n"
            f"Context: {ctx}\n"
            f'Question: {example["question"]}\n'
            "Answer:"
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


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): Title / Context / Question / Answer:
        ctx = example["context"][:_CONTEXT_MAX_CHARS]
        if len(example["context"]) > _CONTEXT_MAX_CHARS:
            ctx += "..."
        return (
            f'Title: {example["title"]}\n'
            f'Context: {ctx}\n'
            f'Question: {example["question"]}\n'
            f'Answer:'
        )


def _to_list(split):
    return [dict(ex) for ex in split]
