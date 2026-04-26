"""Evaluate model accuracy on a task over N samples.

Usage:
  uv run python -m src.scripts.eval_accuracy --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct
  uv run python -m src.scripts.eval_accuracy --task boolq --model Qwen/Qwen2.5-1.5B-Instruct --n 200
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks
from src.utils.seeds import set_seeds


def _resolve_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg == "auto":
        return "bfloat16" if device.startswith("cuda") else "float32"
    return dtype_arg


def run_eval(args):
    set_seeds(args.seed)
    dtype = _resolve_dtype(args.dtype, args.device)

    task = get_task(args.task)
    train_data, _ = task.load_data(train_size=args.n, val_size=0, seed=args.seed)
    examples = train_data[:args.n]
    print(f"[eval] task={args.task}  model={args.model}  n={len(examples)}  dtype={dtype}")

    backend = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    prompts = [task.build_prompt(ex) for ex in examples]

    print(f"[eval] running inference on {len(prompts)} examples...")
    outputs = backend.generate_batch(prompts)
    scores  = [task.score(out, ex) for out, ex in zip(outputs, examples)]

    n_correct = sum(1 for s in scores if s > 0)
    accuracy  = n_correct / len(scores)

    print(f"\n[eval] accuracy: {n_correct}/{len(scores)} = {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Per-example breakdown
    if args.verbose:
        print("\n[eval] per-example results:")
        for i, (ex, out, sc) in enumerate(zip(examples, outputs, scores)):
            label = "✓" if sc > 0 else "✗"
            print(f"  [{i+1:3d}] {label}  output: {out[:80]!r}")

    results = {
        "model":    args.model,
        "task":     args.task,
        "n":        len(scores),
        "seed":     args.seed,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "scores":   scores,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(
            f"results/degen_probe/eval_{args.task}_{args.model.replace('/', '_')}"
            f"_n{len(scores)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] results saved to {out_path}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model accuracy on a task")
    p.add_argument("--task",           required=True, choices=available_tasks())
    p.add_argument("--model",          default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n",              type=int, default=500,
                   help="Number of examples to evaluate")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",          default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output",         default=None)
    p.add_argument("--verbose",        action="store_true",
                   help="Print per-example outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
