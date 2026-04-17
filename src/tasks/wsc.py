from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)

# WSC labels: 0 = no coreference, 1 = coreference


@register("wsc")
class WscTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "wsc")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f'{example["text"]}\n'
            f'Does "{example["span2_text"]}" refer to '
            f'"{example["span1_text"]}"?\n'
            f"Answer yes or no:"
        )

    def build_prompt_base(self, example):
        return (
            "The dog chased the cat until it ran up a tree.\n"
            'Does "it" refer to "cat"?\n'
            "Answer: yes\n\n"
            "The boy gave the toy to his sister, but she didn't want it.\n"
            'Does "she" refer to "boy"?\n'
            "Answer: no\n\n"
            f'{example["text"]}\n'
            f'Does "{example["span2_text"]}" refer to "{example["span1_text"]}"?\n'
            "Answer:"
        )

    def build_prompt_free(self, example):
        return example["text"]

    def predict(self, text: str) -> str | None:
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            return "yes" if yes.start() < no.start() else "no"
        return "yes" if yes else ("no" if no else None)

    def gold_label(self, example: dict) -> str:
        return "yes" if example["label"] == 1 else "no"

    def label_words(self):
        return ["yes", "no"]

    def score_ce(self, log_probs, example):
        # label 1 = coreference (yes), label 0 = no coreference (no)
        correct = "yes" if example["label"] == 1 else "no"
        return log_probs[correct]

    def score(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 1 if yes.start() < no.start() else 0
        elif yes:
            pred = 1
        elif no:
            pred = 0
        else:
            return 0.0  # parse failure
        return 1.0 if pred == example["label"] else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): span2 quoted, span1 unquoted
        return (
            f'{example["text"]}\n'
            f'In the previous sentence, does the pronoun "{example["span2_text"]}" '
            f'refer to {example["span1_text"]}? Yes or No?'
        )

    def label_words_mezo(self):
        return ["Yes", "No"]

    def score_mezo(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 1 if yes.start() < no.start() else 0
        elif yes:
            pred = 1  # coreference
        elif no:
            pred = 0  # no coreference
        else:
            return -1.0
        return 1.0 if pred == example["label"] else -1.0

    def score_ce_mezo(self, log_probs, example):
        correct = "Yes" if example["label"] == 1 else "No"
        return log_probs[correct]


def _to_list(split):
    return [dict(ex) for ex in split]
