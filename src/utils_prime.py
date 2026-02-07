import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from typing import Optional


# IMPORTANT:
# This key must match the argument name that your Wordle/TextArena
# environment uses to read the system prompt.
PROMPT_FIELD = "system_prompt"


@dataclass
class EvalResult:
    reward: float
    metrics: Dict[str, float]
    raw: Dict[str, Any]
    cmd: str


def extract_reward_from_results(results_path: Path, is_jsonl: bool) -> float:
    rewards = []

    if is_jsonl:
        with results_path.open() as f:
            for line in f:
                obj = json.loads(line)
                if "reward" in obj:
                    rewards.append(float(obj["reward"]))
    else:
        data = json.loads(results_path.read_text())
        for obj in data:
            if "reward" in obj:
                rewards.append(float(obj["reward"]))

    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def find_eval_root() -> Path:
    # Look for common eval output locations starting from the current
    # working directory and walking up parent directories. This allows
    # running notebooks from a subfolder (e.g., `notebooks/`) while
    # still locating eval outputs saved at the repository root.
    rel_candidates = [
        Path("evals"),
        Path("prime_evals"),
        Path("environments") / "my_env" / "outputs" / "evals",
    ]

    cwd = Path.cwd()
    for base in [cwd] + list(cwd.parents):
        for rc in rel_candidates:
            p = base / rc
            if p.exists():
                return p

    # fallback to common home locations
    home_candidates = [
        Path.home() / ".prime" / "evals",
        Path.home() / ".cache" / "prime" / "evals",
    ]
    for p in home_candidates:
        if p.exists():
            return p

    raise RuntimeError("Could not locate Prime eval output directory")

def find_latest_results_file(eval_root: Path, start_time: float) -> Path:
    """
    Find the newest results.jsonl or results.json under eval_root.
    Prefers files created after start_time, but falls back to latest overall.
    Safe for repeated calls in training loops.
    """
    candidates = []

    for pattern in ("results.jsonl", "results.json"):
        for p in eval_root.rglob(pattern):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        raise RuntimeError(f"No results.json(l) found under {eval_root}")

    # sort by modification time (newest last)
    candidates.sort(key=lambda p: p.stat().st_mtime)

    # 1) try: newest file AFTER start_time
    for p in reversed(candidates):
        if p.stat().st_mtime >= start_time:
            return p

    # 2) fallback: newest file overall
    return candidates[-1]


def prime_eval(
    env_id: str,
    model: str,
    prompt: str,
    n_episodes: int = 10,
    rollouts_per_example: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    env_args: Optional[Dict[str, Any]] = None,
    extra_cli: Optional[list] = None,
) -> EvalResult:
    """
    Run a Prime evaluation via CLI and return a scalar reward.

    For Wordle/TextArena-style environments, the reward is read from
    `results.jsonl` (per-example `reward` field) and averaged.
    """
    if env_args is None:
        env_args = {}

    # inject prompt
    env_args[PROMPT_FIELD] = prompt

    cmd = [
        "prime",
        "eval",
        "run",
        env_id,
        "-m",
        model,
        "-n",
        str(n_episodes),
        "-r",
        str(rollouts_per_example),
        "-t",
        str(max_tokens),
        "-T",
        str(temperature),
        "-a",
        json.dumps(env_args),
        "-s",  # 🔥 save results to disk
    ]

    if extra_cli:
        cmd += extra_cli

    cmd_str = " ".join(cmd)

    # mark time BEFORE running eval
    start_time = time.time()

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ},
    )

    # locate the newly created eval directory
    eval_root = find_eval_root()

    # 🔥 find newest results file produced by THIS run
    results_path = find_latest_results_file(eval_root, start_time - 5.0)
    run_dir = results_path.parent  # this will be .../model_dir/<run_id>/

    is_jsonl = results_path.name.endswith(".jsonl")

    reward = extract_reward_from_results(results_path, is_jsonl)

    raw = {
        "stdout": proc.stdout,
        "returncode": proc.returncode,
        "run_dir": str(run_dir),
        "results_path": str(results_path),
    }

    return EvalResult(
        reward=reward,
        metrics={},  # Wordle/TextArena: reward already aggregated
        raw=raw,
        cmd=cmd_str,
    )
