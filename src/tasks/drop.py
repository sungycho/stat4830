from __future__ import annotations
import re
import string
from datasets import load_dataset
from src.tasks import Task, register

_PASSAGE_MAX_CHARS = 600


def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return " ".join(text.split())


@register("drop")
class DropTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("ucinlp/drop")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            f"Passage: {passage}\n"
            f'Question: {example["question"]}\n'
            f"Answer in as few words as possible:"
        )

    def build_prompt_base(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            "Passage: The stadium holds 50,000 fans. During the championship game, 43,000 attended.\n"
            "Question: How many seats were empty during the championship game?\n"
            "Answer: 7,000\n\n"
            "Passage: The journey from Paris to London takes about 2 hours and 15 minutes by Eurostar.\n"
            "Question: How long does the train journey take?\n"
            "Answer: 2 hours and 15 minutes\n\n"
            f"Passage: {passage}\n"
            f'Question: {example["question"]}\n'
            "Answer:"
        )

    def score(self, text, example):
        pred = _normalize(text)
        if not pred:
            return -1.0
        spans = example["answers_spans"]["spans"]
        for span in spans:
            if _normalize(span) in pred:
                return 1.0
        return -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): "Passage: <context>\nQuestion: <question>\nAnswer:"
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            f'Passage: {passage}\n'
            f'Question: {example["question"]}\n'
            f'Answer:'
        )


def _to_list(split):
    return [dict(ex) for ex in split]
