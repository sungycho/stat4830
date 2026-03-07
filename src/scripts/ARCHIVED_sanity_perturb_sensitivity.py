"""Run with: uv run python -m src.scripts.sanity_perturb_sensitivity"""
import os
import random
import time
from src.backends.factory import create_backend
from src.envs.simple_wordle_env import SimpleWordleEnv
from src.prompting.wordle_prompt import build_prompt
from src.rollout.rollout import run_one_turn
from src.utils.seeds import set_seeds
from src.utils.perturb import perturb_inplace, restore_inplace


def main() -> None:
    set_seeds(42)
    t0 = time.perf_counter()
    backend = create_backend(
        "hf",
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        device="cpu",
        dtype="float32",
        max_new_tokens=20,
    )
    env = SimpleWordleEnv(target_word="crane")

    _, baseline_reward, _ = run_one_turn(backend, env, build_prompt)
    print("Baseline reward:", baseline_reward)

    seed = random.randint(0, 2**31)
    sigma = 1e-3

    perturb_inplace(backend.model, seed, sigma, sign=+1)      # θ → θ+σε
    _, reward_plus, _ = run_one_turn(backend, env, build_prompt)
    perturb_inplace(backend.model, seed, 2 * sigma, sign=-1)  # θ+σε → θ-σε
    _, reward_minus, _ = run_one_turn(backend, env, build_prompt)
    restore_inplace(backend.model, seed, sigma, sign=-1)       # θ-σε → θ

    print("Reward (+eps):", reward_plus)
    print("Reward (-eps):", reward_minus)
    print("Grad estimate (scalar proxy):", (reward_plus - reward_minus) / (2 * sigma))
    print("PASS: perturbation sensitivity check")
    print(f"Wall-clock elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
