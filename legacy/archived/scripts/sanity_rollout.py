"""Run with: uv run python -m src.scripts.sanity_rollout"""
from src.backends.factory import create_backend
from src.envs.simple_wordle_env import SimpleWordleEnv
from src.prompting.wordle_prompt import build_prompt
from src.rollout.rollout import run_one_turn
from src.utils.seeds import set_seeds


def main() -> None:
    set_seeds(1546)
    backend = create_backend(
        "hf",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
        dtype="float32",
        max_new_tokens=6,
        do_sample=False,
    )
    env = SimpleWordleEnv(target_word="crane")
    text, reward, info = run_one_turn(backend, env, build_prompt)
    print("Model output:", repr(text))
    print("Reward:", reward)
    print("Info:", info)


if __name__ == "__main__":
    main()
