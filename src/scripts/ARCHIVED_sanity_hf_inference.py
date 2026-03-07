"""Run with: uv run python -m src.scripts.sanity_hf_inference"""
import os
from src.backends.factory import create_backend
from src.prompting.wordle_prompt import build_prompt
from src.utils.seeds import set_seeds


def main() -> None:
    set_seeds(42)
    backend = create_backend(
        "hf",
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        device="cpu",
        dtype="float32",
        max_new_tokens=16,
    )
    prompt = build_prompt()
    text1, text2 = backend.generate(prompt), backend.generate(prompt)
    print("GEN1:", repr(text1))
    print("GEN2:", repr(text2))
    assert text1 == text2, "Non-deterministic output"
    print("PASS: determinism check")


if __name__ == "__main__":
    main()
