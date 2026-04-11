"""Countdown task — given a set of numbers and a target, find an arithmetic
expression using those numbers (each exactly once) that equals the target.

Dataset: src/tasks/data/countdown.json (2200 examples from the ES-at-Scale paper).
Paper split: 200 train / 2000 test.

Scoring:
  score()  — binary +1.0 / -1.0 for validation accuracy.
  reward() — composite 0.1 * format_reward + answer_reward (0.0–1.1) for ES
             training, matching the paper's reward_function() exactly.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path

from src.tasks import Task, register

_DATA_PATH = Path(__file__).parent / "data" / "countdown.json"

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.USub: operator.neg,
}

_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/() ]+$")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE  = re.compile(r"<think>.*?</think>",    re.DOTALL)
_FULL_RE   = re.compile(r"^<think>.*?</think>\n<answer>.*?</answer>$", re.DOTALL)


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


def _answer_reward(expr: str, available: list[int], target: int) -> float:
    """Returns 1.0 if expr is valid, uses all numbers exactly, and hits target."""
    if not expr or not _ALLOWED_CHARS.match(expr):
        return 0.0
    used = [int(m) for m in re.findall(r"\b\d+\b", expr)]
    if sorted(used) != sorted(available):
        return 0.0
    result = _safe_eval(expr)
    if result is None:
        return 0.0
    return 1.0 if abs(result - target) < 1e-5 else 0.0


def _format_reward(text: str) -> float:
    """Format reward matching paper's format_reward_function().

    The model output is prefixed with '<think>' before checking, because the
    dataset context already ends with '<think>' and the model continues from there.
    Full format <think>…</think>\\n<answer>…</answer> → 1.0.
    Partial: +0.1 for <think> block, +0.5 for <answer> block.
    """
    full_text = "<think>" + text
    if _FULL_RE.match(full_text):
        return 1.0
    reward = 0.0
    if _THINK_RE.search(full_text):
        reward += 0.1
    if _ANSWER_RE.search(full_text):
        reward += 0.5
    return reward


@register("countdown")
class CountdownTask(Task):

    @property
    def prefer_base_prompt(self) -> bool:
        # Always use the raw context prompt; bypass chat-template wrapping.
        # Matches the paper's completion-mode approach for both base and instruct models.
        return True

    def load_data(self, train_size: int, val_size: int, seed: int):
        # Dataset has 2200 examples. Paper uses 200 train / 2000 val.
        with open(_DATA_PATH) as f:
            data = json.load(f)
        total = train_size + val_size
        if total > len(data):
            raise ValueError(
                f"Requested {total} examples but countdown.json only has {len(data)}."
            )
        # Normalize field name: 'numbers' → 'nums'
        def _norm(item: dict) -> dict:
            return {
                "nums":     item["numbers"],
                "target":   float(item["target"]),
                "context":  item["context"],
                "solution": item.get("solution", ""),
            }
        train = [_norm(data[i]) for i in range(train_size)]
        val   = [_norm(data[i]) for i in range(train_size, train_size + val_size)]
        return train, val

    def build_prompt_base(self, example: dict) -> str:
        # Return the pre-formatted prompt from the dataset directly.
        # It ends with '\n<think>' so the model continues inside the think block.
        return example["context"]

    def build_prompt(self, example: dict) -> str:
        # Clean user-message format for chat-template backends.
        # Not used when prefer_base_prompt is True (which it always is for this task).
        nums   = example["nums"]
        target = example["target"]
        nums_str = " ".join(str(n) for n in nums)
        return (
            f"Using the numbers [{nums_str}], create an equation that equals {target}. "
            f"You can use basic arithmetic operations (+, -, *, /) and each number can "
            f"only be used once. Show your work in <think> </think> tags. And return the "
            f"final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
        )

    def score(self, text: str, example: dict) -> float:
        """Binary accuracy: +1.0 if answer is correct, -1.0 otherwise.

        Matches paper's answer_reward_function() strictness:
        - Must have <answer> tags (no fallback extraction)
        - Allowed chars only: digits, +, -, *, /, (, ), space
        - Must use ALL provided numbers exactly (sorted match)
        """
        target    = float(example["target"])
        available = [int(n) for n in example["nums"]]

        matches = _ANSWER_RE.findall(text)
        if not matches:
            return -1.0
        expr = matches[-1].strip()

        return 1.0 if _answer_reward(expr, available, target) == 1.0 else -1.0

    def reward(self, text: str, example: dict) -> float:
        """Composite training reward matching paper's reward_function().

        Total = 0.1 * format_reward + answer_reward, range [0.0, 1.1].
        """
        target    = float(example["target"])
        available = [int(n) for n in example["nums"]]

        fmt = _format_reward(text)

        matches = _ANSWER_RE.findall(text)
        ans = _answer_reward(matches[-1].strip(), available, target) if matches else 0.0

        return 0.1 * fmt + ans
