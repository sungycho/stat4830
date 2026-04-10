"""Countdown task — given a set of numbers and a target, find an arithmetic
expression using those numbers (each at most once) that equals the target.

Dataset: Jiayi-Pan/Countdown-Tasks-3to4 (HuggingFace)
Scoring: +1 if the model's expression evaluates to the target using only the
         given numbers, -1 otherwise.
"""
from __future__ import annotations

import ast
import operator
import re
from datasets import load_dataset
from src.tasks import Task, register

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression safely (no eval())."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    try:
        return _eval_node(tree.body)
    except Exception:
        return None


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return _SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        operand = _eval_node(node.operand)
        return None if operand is None else _SAFE_OPS[type(node.op)](operand)
    return None


def _extract_numbers_used(expr: str) -> list[int]:
    """Pull out all integer literals from an expression string."""
    return [int(m) for m in re.findall(r"\b\d+\b", expr)]


def _numbers_valid(used: list[int], available: list[int]) -> bool:
    """Check used numbers are a multiset-subset of available."""
    avail = list(available)
    for n in used:
        if n not in avail:
            return False
        avail.remove(n)
    return True


def _extract_expression(text: str) -> str | None:
    """Pull the first plausible arithmetic expression from model output."""
    # Try to find "= <number>" pattern and work backward
    eq_match = re.search(r"=\s*([\d]+)", text)
    # Look for lines containing digits and operators
    for line in text.splitlines():
        line = line.strip()
        if re.search(r"\d", line) and re.search(r"[+\-*/]", line):
            # Strip trailing "= <answer>" if present
            line = re.sub(r"\s*=\s*[\d]+\s*$", "", line).strip()
            return line
    return None


@register("countdown")
class CountdownTask(Task):
    def load_data(self, train_size: int, val_size: int, seed: int):
        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        ds = ds.shuffle(seed=seed)
        total = train_size + val_size
        ds = ds.select(range(min(total, len(ds))))
        train = [dict(ex) for ex in ds.select(range(train_size))]
        val   = [dict(ex) for ex in ds.select(range(train_size, train_size + val_size))]
        return train, val

    def build_prompt(self, example: dict) -> str:
        nums   = example["nums"]
        target = example["target"]
        nums_str = ", ".join(str(n) for n in nums)
        return (
            f"Using the numbers {nums_str}, create an arithmetic expression "
            f"(using +, -, *, / and each number at most once) that equals {target}.\n"
            f"Write only the expression, nothing else."
        )

    def build_prompt_base(self, example: dict) -> str:
        nums   = example["nums"]
        target = example["target"]
        nums_str = ", ".join(str(n) for n in nums)
        return (
            "Using the numbers 3, 8, 5, create an arithmetic expression "
            "(using +, -, *, / and each number at most once) that equals 24.\n"
            "Expression: 3 * 8\n\n"
            "Using the numbers 2, 7, 9, create an arithmetic expression "
            "(using +, -, *, / and each number at most once) that equals 18.\n"
            "Expression: 9 + 7 + 2\n\n"
            f"Using the numbers {nums_str}, create an arithmetic expression "
            f"(using +, -, *, / and each number at most once) that equals {target}.\n"
            "Expression:"
        )

    def score(self, text: str, example: dict) -> float:
        target    = int(example["target"])
        available = [int(n) for n in example["nums"]]

        expr = _extract_expression(text)
        if expr is None:
            return -1.0

        used = _extract_numbers_used(expr)
        if not _numbers_valid(used, available):
            return -1.0

        result = _safe_eval(expr)
        if result is None:
            return -1.0

        return 1.0 if abs(result - target) < 1e-6 else -1.0
