from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_ENT = re.compile(r"\bentailment\b", re.IGNORECASE)
_NEU = re.compile(r"\bneutral\b", re.IGNORECASE)
_CON = re.compile(r"\bcontradiction\b", re.IGNORECASE)

# MNLI labels: 0=entailment, 1=neutral, 2=contradiction
_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


@register("mnli")
class MnliTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("glue", "mnli")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        val_pool = ds["validation_matched"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f'Premise: "{example["premise"]}"\n'
            f'Hypothesis: "{example["hypothesis"]}"\n'
            f"Does the premise entail the hypothesis, "
            f"contradict it, or is it neutral?\n"
            f"Answer entailment, contradiction, or neutral:"
        )

    def build_prompt_base(self, example):
        return (
            'Premise: "The woman is playing tennis outdoors."\n'
            'Hypothesis: "A person is engaged in a sport."\n'
            "Answer: entailment\n\n"
            'Premise: "Two men are cooking steaks on a grill."\n'
            'Hypothesis: "The men are vegetarians."\n'
            "Answer: contradiction\n\n"
            'Premise: "She bought a new dress for the event."\n'
            'Hypothesis: "She spent more than fifty dollars."\n'
            "Answer: neutral\n\n"
            f'Premise: "{example["premise"]}"\n'
            f'Hypothesis: "{example["hypothesis"]}"\n'
            "Answer:"
        )

    def predict(self, text: str) -> str | None:
        ent = _ENT.search(text)
        neu = _NEU.search(text)
        con = _CON.search(text)
        hits = [(m.start(), lbl) for m, lbl in [(ent, "entailment"), (neu, "neutral"), (con, "contradiction")] if m]
        return min(hits, key=lambda h: h[0])[1] if hits else None

    def gold_label(self, example: dict) -> str:
        return _LABEL_MAP[example["label"]]

    def label_words(self):
        return ["entailment", "neutral", "contradiction"]

    def score_ce(self, log_probs, example):
        correct = _LABEL_MAP[example["label"]]
        return log_probs[correct]

    def score(self, text, example):
        ent = _ENT.search(text)
        neu = _NEU.search(text)
        con = _CON.search(text)
        hits = [
            (ent.start(), "entailment") if ent else None,
            (neu.start(), "neutral") if neu else None,
            (con.start(), "contradiction") if con else None,
        ]
        hits = [h for h in hits if h is not None]
        if not hits:
            return 0.0  # parse failure
        pred = min(hits, key=lambda h: h[0])[1]
        gold = _LABEL_MAP[example["label"]]
        return 1.0 if pred == gold else -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
