"""Rho probe sweep: run probe_degeneracy for every task × model combination.

Prompt/template decisions:
  Base models  (OPT, Llama, Qwen-Math) : --prompt-style complex --no-chat-template
  Instruct models (Qwen-Instruct)       : --prompt-style simple  (chat template auto-detected)

OPT-66B warning: ~132 GB in bfloat16 — will OOM on a single 80 GB GPU.
  Pass --skip-models opt-66b to exclude it, or ensure device_map support is added first.

Usage:
  uv run python -m src.scripts.adhoc.run_rho_sweep
  uv run python -m src.scripts.adhoc.run_rho_sweep --skip-done
  uv run python -m src.scripts.adhoc.run_rho_sweep --models opt-350m opt-1.3b --tasks sst2 rte
  uv run python -m src.scripts.adhoc.run_rho_sweep --skip-models opt-66b --K 200
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict] = [
    # --- OPT (base) ---
    dict(name="opt-350m",              hf="facebook/opt-350m",               instruct=False, batch_size=16),
    dict(name="opt-1.3b",              hf="facebook/opt-1.3b",               instruct=False, batch_size=16),
    dict(name="opt-2.7b",              hf="facebook/opt-2.7b",               instruct=False, batch_size=16),
    dict(name="opt-13b",               hf="facebook/opt-13b",                instruct=False, batch_size=16),
    dict(name="opt-66b",               hf="facebook/opt-66b",                instruct=False, batch_size=4),
    # --- Llama (base) ---
    dict(name="llama2-7b",             hf="meta-llama/Llama-2-7b-hf",        instruct=False, batch_size=16),
    dict(name="llama2-13b",            hf="meta-llama/Llama-2-13b-hf",       instruct=False, batch_size=16),
    dict(name="llama3-8b",             hf="meta-llama/Meta-Llama-3-8B",      instruct=False, batch_size=16),
    dict(name="llama3.1-8b",           hf="meta-llama/Meta-Llama-3.1-8B",    instruct=False, batch_size=16),
    dict(name="llama3.2-1b",           hf="meta-llama/Llama-3.2-1B",         instruct=False, batch_size=16),
    dict(name="llama3.2-3b",           hf="meta-llama/Llama-3.2-3B",         instruct=False, batch_size=16),
    # --- Qwen-Math (base) ---
    dict(name="qwen2.5-math-1.5b",     hf="Qwen/Qwen2.5-Math-1.5B",         instruct=False, batch_size=16),
    dict(name="qwen2.5-math-7b",       hf="Qwen/Qwen2.5-Math-7B",           instruct=False, batch_size=16),
    # --- Qwen-Instruct ---
    dict(name="qwen2.5-1.5b-instruct", hf="Qwen/Qwen2.5-1.5B-Instruct",     instruct=True,  batch_size=16),
    dict(name="qwen2.5-3b-instruct",   hf="Qwen/Qwen2.5-3B-Instruct",       instruct=True,  batch_size=16),
    dict(name="qwen2.5-7b-instruct",   hf="Qwen/Qwen2.5-7B-Instruct",       instruct=True,  batch_size=16),
]

# ---------------------------------------------------------------------------
# Task registry: max_new_tokens per task
# Classification tasks need only a label word (4 tokens).
# Generation tasks need enough tokens to express an answer.
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, int] = {
    "sst2":      4,
    "sst5":      4,
    "rte":       4,
    "boolq":     4,
    "mnli":      4,
    "cb":        4,
    "wsc":       4,
    "wic":       4,
    "copa":      8,
    "trec":      4,
    "squad":     32,
    "drop":      8,
    "record":    16,
    "gsm8k":     256,
    "math500":   256,
    "countdown": 256,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hf_token() -> str | None:
    """Read HF_TOKEN from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN"):
                _, _, val = line.partition("=")
                return val.strip().strip('"').strip("'")
    return None


def _build_cmd(model: dict, task: str, max_new_tokens: int, K: int, out_path: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.scripts.adhoc.probe_degeneracy",
        "--task",           task,
        "--model",          model["hf"],
        "--K",              str(K),
        "--batch-size",     str(model["batch_size"]),
        "--max-new-tokens", str(max_new_tokens),
        "--sigma",          "0.001",
        "--output",         str(out_path),
    ]
    if model["instruct"]:
        cmd += ["--prompt-style", "simple"]
    else:
        cmd += ["--prompt-style", "complex", "--no-chat-template"]
    return cmd


def _run(cmd: list[str], env: dict) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rho probe sweep: all tasks × models")
    p.add_argument("--models",       nargs="*", default=None,
                   help="Subset of model names to run (default: all)")
    p.add_argument("--skip-models",  nargs="*", default=None,
                   help="Model names to skip (e.g. --skip-models opt-66b)")
    p.add_argument("--tasks",        nargs="*", default=None,
                   help="Subset of tasks to run (default: all)")
    p.add_argument("--K",            type=int,  default=100,
                   help="Probe pairs per run (default 100 → n=1600 with B=16, ±0.05 precision)")
    p.add_argument("--skip-done",    action="store_true", default=False,
                   help="Skip runs whose output file already exists")
    p.add_argument("--out-dir",      default="results/rho_sweep",
                   help="Root output directory (default: results/rho_sweep)")
    return p.parse_args()


def main():
    args = parse_args()

    # Filter models
    models = MODEL_REGISTRY
    if args.models:
        models = [m for m in models if m["name"] in args.models]
    if args.skip_models:
        models = [m for m in models if m["name"] not in args.skip_models]

    # Filter tasks
    tasks = dict(TASK_REGISTRY)
    if args.tasks:
        tasks = {t: v for t, v in tasks.items() if t in args.tasks}

    out_root = Path(args.out_dir)
    log_path = out_root / f"sweep_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    # HuggingFace token
    hf_token = _load_hf_token()
    env = {**os.environ}
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("[warn] HF_TOKEN not found — gated models (Llama-2, Llama-3) may fail")

    total = len(models) * len(tasks)
    done = skipped = failed = 0
    print(f"\nRho sweep: {len(models)} models × {len(tasks)} tasks = {total} runs")
    print(f"K={args.K}  skip_done={args.skip_done}  out={out_root}\n")

    for model in models:
        model_dir = out_root / model["name"]
        model_dir.mkdir(parents=True, exist_ok=True)

        for task, max_new_tokens in tasks.items():
            out_path = model_dir / f"{task}.json"
            run_num = done + skipped + failed + 1

            print(f"[{run_num}/{total}] {model['name']} × {task}", end="  ")

            if args.skip_done and out_path.exists():
                print("SKIP (exists)")
                skipped += 1
                continue

            print()
            cmd = _build_cmd(model, task, max_new_tokens, args.K, out_path)
            rc = _run(cmd, env)

            entry = {
                "model":     model["name"],
                "task":      task,
                "rc":        rc,
                "out_path":  str(out_path),
                "timestamp": datetime.now().isoformat(),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if rc == 0:
                done += 1
                print(f"  -> OK  ({out_path})")
            else:
                failed += 1
                print(f"  -> FAILED (rc={rc})")

    print(f"\nDone: {done} OK  |  {skipped} skipped  |  {failed} failed")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
