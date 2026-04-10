from __future__ import annotations
import re
from datasets import load_dataset
from src.tasks import Task, register

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBERS = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content from text."""
    matches = _BOXED.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_answer(ans: str) -> str:
    """Normalize for comparison: strip, lowercase, remove $, commas."""
    ans = ans.strip().lower()
    ans = ans.replace("$", "").replace(",", "")
    ans = ans.replace("\\!", "").replace("\\,", "")
    ans = re.sub(r"\s+", " ", ans)
    return ans


def _try_numeric(s: str) -> float | None:
    """Try to parse as a number for numeric comparison."""
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


@register("math500")
class Math500Task(Task):
    def load_data(self, train_size, val_size, seed):
        ds = load_dataset("HuggingFaceH4/MATH-500")
        # MATH-500 has only a test split
        test = ds["test"].shuffle(seed=seed)
        total = min(train_size + val_size, len(test))
        sel = test.select(range(total))
        train = [dict(ex) for ex in sel.select(
            range(min(train_size, total))
        )]
        val = [dict(ex) for ex in sel.select(
            range(min(train_size, total), total)
        )]
        return train, val

    def build_prompt(self, example):
        return (
            f'{example["problem"]}\n'
            f"Solve the problem. Put your final answer "
            f"in \\boxed{{}}."
        )

    def build_prompt_base(self, example):
        return (
            "Problem: What is the value of $2 + 2$?\n"
            "Solution: $2 + 2 = 4$. \\boxed{4}\n\n"
            "Problem: Simplify $(3 \\times 4) - 5$.\n"
            "Solution: $3 \\times 4 = 12$, then $12 - 5 = 7$. \\boxed{7}\n\n"
            f'Problem: {example["problem"]}\n'
            "Solution:"
        )

    def score(self, text, example):
        gold_boxed = _extract_boxed(example["solution"])
        if gold_boxed is None:
            return -1.0

        gold_norm = _normalize_answer(gold_boxed)

        # Try to find \\boxed{} in model output first
        pred_boxed = _extract_boxed(text)
        if pred_boxed is not None:
            pred_norm = _normalize_answer(pred_boxed)
            if pred_norm == gold_norm:
                return 1.0
            g, p = _try_numeric(gold_norm), _try_numeric(
                pred_norm
            )
            if g is not None and p is not None:
                if abs(g - p) < 1e-3:
                    return 1.0
            return -1.0

        # Fallback: check if last number in output matches
        nums = _NUMBERS.findall(text)
        if nums:
            pred_num = _try_numeric(nums[-1])
            gold_num = _try_numeric(gold_norm)
            if (
                pred_num is not None
                and gold_num is not None
                and abs(pred_num - gold_num) < 1e-3
            ):
                return 1.0

        return -1.0
