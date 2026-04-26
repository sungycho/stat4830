from __future__ import annotations


def build_prompt() -> str:
    return (
        "You are playing a one-turn Wordle-style game.\n"
        "Your task is to guess exactly one 5-letter English word.\n"
        "Format: respond with the word in square brackets only.\n"
        "Example response: [crane]\n"
        "Rules: lowercase letters, exactly five characters, no other text."
    )
