from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)

# RTE labels: 0 = entailment (yes), 1 = not_entailment (no)


@register("rte")
class RteTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("glue", "rte")
        train = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
        val_pool = ds["validation"]
        val = val_pool.shuffle(seed=seed).select(range(min(val_size, len(val_pool))))
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f'Premise: "{example["sentence1"]}"\n'
            f'Hypothesis: "{example["sentence2"]}"\n'
            f"Does the hypothesis follow from the premise? Answer yes or no:"
        )

    def build_prompt_base(self, example):
        return (
            'Premise: "The Eiffel Tower is located in Paris, France."\n'
            'Hypothesis: "The Eiffel Tower is in France."\n'
            "Answer: yes\n\n"
            'Premise: "The cat sat on the mat."\n'
            'Hypothesis: "The dog sat on the mat."\n'
            "Answer: no\n\n"
            f'Premise: "{example["sentence1"]}"\n'
            f'Hypothesis: "{example["sentence2"]}"\n'
            "Answer:"
        )

    def predict(self, text: str) -> str | None:
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            return "yes" if yes.start() < no.start() else "no"
        return "yes" if yes else ("no" if no else None)

    def gold_label(self, example: dict) -> str:
        return "yes" if example["label"] == 0 else "no"

    def label_words(self):
        return ["yes", "no"]

    def score_ce(self, log_probs, example):
        # label 0 = entailment (yes), label 1 = not_entailment (no)
        correct = "yes" if example["label"] == 0 else "no"
        return log_probs[correct]

    def score(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 0 if yes.start() < no.start() else 1
        elif yes:
            pred = 0  # entailment
        elif no:
            pred = 1  # not_entailment
        else:
            return 0.0  # parse failure
        return 1.0 if pred == example["label"] else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): premise then yes/no question with hypothesis in quotes
        return (
            f'{example["sentence1"]}\n'
            f'Does this mean that "{example["sentence2"]}" is true? Yes or No?'
        )

    def label_words_mezo(self):
        return ["Yes", "No"]

    def score_mezo(self, text, example):
        yes = _YES.search(text)
        no = _NO.search(text)
        if yes and no:
            pred = 0 if yes.start() < no.start() else 1
        elif yes:
            pred = 0  # entailment
        elif no:
            pred = 1  # not_entailment
        else:
            return -1.0
        return 1.0 if pred == example["label"] else -1.0

    def score_ce_mezo(self, log_probs, example):
        correct = "Yes" if example["label"] == 0 else "No"
        return log_probs[correct]


def _to_list(split):
    return [dict(ex) for ex in split]
