from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_ENT = re.compile(r"\bentailment\b", re.IGNORECASE)
_CON = re.compile(r"\bcontradiction\b", re.IGNORECASE)
_NEU = re.compile(r"\bneutral\b", re.IGNORECASE)

# CB (SuperGLUE) labels: 0=entailment, 1=contradiction, 2=neutral
_LABEL_MAP = {0: "entailment", 1: "contradiction", 2: "neutral"}

_YES_M = re.compile(r"\byes\b", re.IGNORECASE)
_NO_M  = re.compile(r"\bno\b",  re.IGNORECASE)
_MAY_M = re.compile(r"\bmaybe\b", re.IGNORECASE)
# MeZO: entailment→Yes, contradiction→No, neutral→Maybe
_MEZO_LABEL_MAP = {0: "Yes", 1: "No", 2: "Maybe"}


@register("cb")
class CbTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("super_glue", "cb")
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
            f'Premise: "{example["premise"]}"\n'
            f'Hypothesis: "{example["hypothesis"]}"\n'
            f"Does the premise entail the hypothesis, "
            f"contradict it, or is it neutral?\n"
            f"Answer entailment, contradiction, or neutral:"
        )

    def build_prompt_base(self, example):
        return (
            'Premise: "John said that he would come to the party tomorrow."\n'
            'Hypothesis: "John will attend the party."\n'
            "Answer: entailment\n\n"
            'Premise: "Mary told us she hates cold weather."\n'
            'Hypothesis: "Mary enjoys the cold."\n'
            "Answer: contradiction\n\n"
            'Premise: "The experiment was conducted last Tuesday."\n'
            'Hypothesis: "The scientists worked on Wednesday."\n'
            "Answer: neutral\n\n"
            f'Premise: "{example["premise"]}"\n'
            f'Hypothesis: "{example["hypothesis"]}"\n'
            "Answer:"
        )

    def label_words(self):
        return ["entailment", "contradiction", "neutral"]

    def score_ce(self, log_probs, example):
        correct = _LABEL_MAP[example["label"]]
        return log_probs[correct]

    def score(self, text, example):
        ent = _ENT.search(text)
        con = _CON.search(text)
        neu = _NEU.search(text)
        hits = [
            (ent.start(), "entailment") if ent else None,
            (con.start(), "contradiction") if con else None,
            (neu.start(), "neutral") if neu else None,
        ]
        hits = [h for h in hits if h is not None]
        if not hits:
            return -1.0
        pred = min(hits, key=lambda h: h[0])[1]
        gold = _LABEL_MAP[example["label"]]
        return 1.0 if pred == gold else -1.0


    def build_prompt_mezo(self, example):
        # MeZO paper (Table 14): "Suppose <premise> Can we infer that '<hypothesis>'? Yes, No, or Maybe?"
        return (
            f'Suppose {example["premise"]} '
            f'Can we infer that "{example["hypothesis"]}"? Yes, No, or Maybe?'
        )

    def label_words_mezo(self):
        # MeZO maps: entailment→Yes, contradiction→No, neutral→Maybe
        return ["Yes", "No", "Maybe"]

    def score_mezo(self, text, example):
        yes = _YES_M.search(text)
        no  = _NO_M.search(text)
        may = _MAY_M.search(text)
        hits = [
            (yes.start(), 0) if yes else None,  # 0=entailment
            (no.start(),  1) if no  else None,  # 1=contradiction
            (may.start(), 2) if may else None,  # 2=neutral
        ]
        hits = [h for h in hits if h is not None]
        if not hits:
            return -1.0
        pred = min(hits, key=lambda h: h[0])[1]
        return 1.0 if pred == example["label"] else -1.0

    def score_ce_mezo(self, log_probs, example):
        correct = _MEZO_LABEL_MAP[example["label"]]
        return log_probs[correct]


def _to_list(split):
    return [dict(ex) for ex in split]
