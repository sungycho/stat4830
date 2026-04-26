from __future__ import annotations
import re

_PATTERN = re.compile(r"\[([a-zA-Z]{5})\]")


def parse_guess(text: str) -> str | None:
    match = _PATTERN.search(text)
    return match.group(1).lower() if match else None
