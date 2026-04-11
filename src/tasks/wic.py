from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)

# WIC labels: 0 = different sense, 1 = same sense


@register("wic")
class WicTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "wic")
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
            f'Sentence 1: "{example["sentence1"]}"\n'
            f'Sentence 2: "{example["sentence2"]}"\n'
            f'Is the word "{example["word"]}" used in the '
            f"same sense in both sentences?\n"
            f"Answer yes or no:"
        )

    def build_prompt_base(self, example):
        return (
            'Sentence 1: "He wound a thread around the spool."\n'
            'Sentence 2: "The nurse wound the bandage around his arm."\n'
            'Is the word "wound" used in the same sense in both sentences?\n'
            "Answer: yes\n\n"
            'Sentence 1: "She went to the bank to deposit money."\n'
            'Sentence 2: "They picnicked on the river bank."\n'
            'Is the word "bank" used in the same sense in both sentences?\n'
            "Answer: no\n\n"
            f'Sentence 1: "{example["sentence1"]}"\n'
            f'Sentence 2: "{example["sentence2"]}"\n'
            f'Is the word "{example["word"]}" used in the same sense in both sentences?\n'
            "Answer:"
        )

    def label_words(self):
        return ["yes", "no"]

    def score_ce(self, log_probs, example):
        # label 1 = same sense (yes), label 0 = different sense (no)
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
            return -1.0
        return 1.0 if pred == example["label"] else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): question first, then both sentences
        return (
            f'Does the word "{example["word"]}" have the same meaning in these two sentences? '
            f'Yes, No?\n'
            f'{example["sentence1"]}\n'
            f'{example["sentence2"]}'
        )


def _to_list(split):
    return [dict(ex) for ex in split]
