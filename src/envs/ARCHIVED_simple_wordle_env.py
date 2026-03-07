from __future__ import annotations
from src.parsing.bracket_parser import parse_guess


class SimpleWordleEnv:
    def __init__(self, target_word: str = "crane"):
        self.target = target_word.lower()

    def reset(self) -> str:
        return ""

    def step(self, guess_text: str) -> tuple[None, float, bool, dict]:
        guess = parse_guess(guess_text)
        if guess is None:
            return None, -1.0, True, {"error": "invalid_format"}
        if guess == self.target:
            return None, 1.0, True, {"correct": True}
        greens = sum(g == t for g, t in zip(guess, self.target))
        return None, 0.2 * greens, True, {"correct": False}
