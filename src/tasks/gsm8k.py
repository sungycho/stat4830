from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_FINAL_NUMBER = re.compile(r"####\s*([\d,]+(?:\.\d+)?)")
_NUMBERS = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _parse_number(s: str) -> float | None:
    """Parse a number string, stripping commas."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _extract_gold_answer(answer_str: str) -> float | None:
    """Extract the number after #### in the GSM8K answer."""
    m = _FINAL_NUMBER.search(answer_str)
    if m:
        return _parse_number(m.group(1))
    return None


def _extract_pred_number(text: str) -> float | None:
    """Extract the last number from model output."""
    matches = _NUMBERS.findall(text)
    if matches:
        return _parse_number(matches[-1])
    return None


@register("gsm8k")
class Gsm8kTask(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("openai/gsm8k", "main")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(train_size, len(ds["train"])))
        )
        # GSM8K has no validation split; use test
        val_pool = ds["test"]
        val = val_pool.shuffle(seed=seed).select(
            range(min(val_size, len(val_pool)))
        )
        return _to_list(train), _to_list(val)

    def build_prompt(self, example):
        return (
            f'{example["question"]}\n'
            f"Solve step by step. "
            f"End with the final numeric answer after ####."
        )

    def score(self, text, example):
        gold = _extract_gold_answer(example["answer"])
        if gold is None:
            return -1.0
        pred = _extract_pred_number(text)
        if pred is None:
            return -1.0
        return 1.0 if abs(pred - gold) < 1e-3 else -1.0


def _to_list(split):
    return [dict(ex) for ex in split]
