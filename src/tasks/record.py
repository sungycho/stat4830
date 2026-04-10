from __future__ import annotations
from datasets import load_dataset
from src.tasks import Task, register

_PASSAGE_MAX_CHARS = 600


@register("record")
class RecordTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "record")
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
            f'Query: {example["query"]}\n'
            f"What entity replaces @placeholder? "
            f"Answer with the entity name only:"
        )

    def build_prompt_base(self, example):
        passage = example["passage"][:_PASSAGE_MAX_CHARS]
        if len(example["passage"]) > _PASSAGE_MAX_CHARS:
            passage += "..."
        return (
            "Passage: Clark was born in London and later moved to New York to pursue his career.\n"
            "Query: @placeholder is where Clark was born.\n"
            "Entity: London\n\n"
            "Passage: The 2012 Summer Olympics were held in London at the newly built Olympic Stadium.\n"
            "Query: The venue for the 2012 Olympics was @placeholder.\n"
            "Entity: Olympic Stadium\n\n"
            f"Passage: {passage}\n"
            f'Query: {example["query"]}\n'
            "Entity:"
        )

    def score(self, text, example):
        pred = text.strip().lower()
        if not pred:
            return -1.0
        for ans in example["answers"]:
            if ans.lower() in pred:
                return 1.0
        return -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
