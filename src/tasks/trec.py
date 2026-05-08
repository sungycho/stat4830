from __future__ import annotations
import random
import re
import urllib.request
from pathlib import Path
from src.tasks import Task, register

_LABELS = ["abbreviation", "entity", "description", "human", "location", "numeric"]
_COARSE_MAP = {"ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5}
_PATTERNS = {label: re.compile(rf"\b{label}\b", re.IGNORECASE) for label in _LABELS}

_TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
_TEST_URL  = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
_CACHE_DIR = Path.home() / ".cache" / "stat4830_trec"


def _fetch_trec(url: str, cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    examples = []
    with open(cache_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_part, _, text = line.partition(" ")
            coarse = label_part.partition(":")[0]
            label = _COARSE_MAP.get(coarse, -1)
            if label >= 0:
                examples.append({"text": text, "label": label})
    return examples


@register("trec")
class TrecTask(Task):
    def load_data(self, train_size, val_size, seed):
        train_raw = _fetch_trec(_TRAIN_URL, _CACHE_DIR / "train.label")
        test_raw  = _fetch_trec(_TEST_URL,  _CACHE_DIR / "test.label")
        rng = random.Random(seed)
        rng.shuffle(train_raw)
        rng.shuffle(test_raw)
        return train_raw[:min(train_size, len(train_raw))], test_raw[:min(val_size, len(test_raw))]

    def build_prompt(self, example):
        return (
            f"Classify the question into one of these types: "
            f"abbreviation, entity, description, human, location, numeric.\n"
            f"Question: {example['text']}\n"
            f"Type:"
        )

    def label_words(self):
        return _LABELS

    def score_ce(self, log_probs, example):
        correct = _LABELS[example["label"]]
        return log_probs[correct]

    def score(self, text, example):
        first_match = None
        first_pos = len(text) + 1
        for label, pat in _PATTERNS.items():
            m = pat.search(text)
            if m and m.start() < first_pos:
                first_match = label
                first_pos = m.start()
        if first_match is None:
            return -1.0
        pred = _LABELS.index(first_match)
        return 1.0 if pred == example["label"] else -1.0
