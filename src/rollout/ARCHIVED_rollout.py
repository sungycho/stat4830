from __future__ import annotations
from typing import Callable


def run_one_turn(backend, env, prompt_builder: Callable[[], str]) -> tuple[str, float, dict]:
    env.reset()
    prompt = prompt_builder()
    text = backend.generate(prompt)
    _, reward, _, info = env.step(text)
    return text, reward, info
