# src/scripts/sanity_hf_wordle.py
import os
import random
import numpy as np
import torch

from src.backends.hf_backend import HFBackend


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you later run on GPU:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    seed = 0
    set_seeds(seed)

    # Start tiny + fast
    model_name = os.environ.get("MODEL_NAME", "distilgpt2")

    backend = HFBackend(
        model_name=model_name,
        device="cpu",          # keep CPU for local sanity; GPU later
        dtype="float32",
        max_new_tokens=16,
    )

    # Replace this prompt with your actual Wordle policy prompt template
    prompt = "You are playing Wordle. Reply with one 5-letter guess only."

    # 1) Basic generation
    text1 = backend.generate(prompt)
    text2 = backend.generate(prompt)

    print("GEN1:", repr(text1))
    print("GEN2:", repr(text2))
    print("Deterministic match:", text1 == text2)

    # 2) Hook into your environment (pseudo-code)
    # If you already have a Wordle env class, plug it in here.
    #
    # env = WordleEnv(seed=seed, target_word="crane")  # example
    # action = parse_guess(text1)                      # your parser
    # obs, reward, done, info = env.step(action)
    # print("Reward:", reward, "Done:", done)

    print("\nNext: plug in your env + reward call where indicated.")


if __name__ == "__main__":
    main()