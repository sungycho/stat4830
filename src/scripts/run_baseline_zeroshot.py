"""Zero-shot baseline runner: evaluate pre-trained models on tasks
without any fine-tuning to establish baseline accuracy for each
model/task combination.

Models and tasks are drawn from the MeZO Scaling Laws Experiment
tracker.

Usage
-----
  # Single model, single task
  uv run python -m src.scripts.run_baseline_zeroshot \\
      --models opt-350m --tasks boolq --device cuda

  # Multiple models x multiple tasks
  uv run python -m src.scripts.run_baseline_zeroshot \\
      --models opt-350m opt-1.3b --tasks sst2 boolq rte \\
      --device cuda

  # All registered models on all available tasks
  uv run python -m src.scripts.run_baseline_zeroshot \\
      --all-models --all-tasks --device cuda

  # List available models / tasks
  uv run python -m src.scripts.run_baseline_zeroshot --list
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks


# ------------------------------------------------------------------
# Model registry: short alias -> HuggingFace model ID
# Sourced from MeZO Scaling Laws Experiment spreadsheet.
# ------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    # OPT family
    "opt-350m":        "facebook/opt-350m",
    "opt-1.3b":        "facebook/opt-1.3b",
    "opt-2.7b":        "facebook/opt-2.7b",
    "opt-13b":         "facebook/opt-13b",
    "opt-66b":         "facebook/opt-66b",
    # LLaMA family
    "llama-2-7b":      "meta-llama/Llama-2-7b-hf",
    "llama-3-8b":      "meta-llama/Meta-Llama-3-8B",
    "llama-2-13b":     "meta-llama/Llama-2-13b-hf",
    "llama-2-70b":     "meta-llama/Llama-2-70b-hf",
    "llama-3.1-70b":   "meta-llama/Llama-3.1-70B",
    # GPT-2
    "gpt2-xl":         "gpt2-xl",
    # Phi
    "phi-2":           "microsoft/phi-2",
    # Qwen 2.5 Instruct
    "qwen-1.5b":       "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":         "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b":         "Qwen/Qwen2.5-7B-Instruct",
    # Qwen 2.5 Math
    "qwen-1.5b-math":  "Qwen/Qwen2.5-Math-1.5B",
    "qwen-7b-math":    "Qwen/Qwen2.5-Math-7B",
}

# Default max_new_tokens per task
_TASK_MAX_TOKENS: dict[str, int] = {
    "sst2":      4,
    "boolq":     4,
    "rte":       4,
    "mnli":      4,
    "cb":        4,
    "wsc":       4,
    "wic":       4,
    "copa":      4,
    "squad":     32,
    "record":    32,
    "drop":      32,
    "gsm8k":     512,
    "math500":   512,
    "countdown": 256,
}
_DEFAULT_MAX_TOKENS = 8


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(
    backend, val_data: list[dict], task, batch_size: int,
) -> dict:
    """Run zero-shot eval and return detailed results."""
    correct = 0
    total = len(val_data)
    parse_failures = 0

    for i in range(0, total, batch_size):
        chunk = val_data[i:i + batch_size]
        prompts = [task.build_prompt(ex) for ex in chunk]
        outputs = backend.generate_batch(prompts)
        for text, ex in zip(outputs, chunk):
            score = task.score(text, ex)
            if score > 0:
                correct += 1
            elif score < -0.5:
                parse_failures += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
    }


def resolve_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg == "auto":
        if device.startswith("cuda"):
            return "bfloat16"
        return "float32"
    return dtype_arg


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Zero-shot baseline evaluation across models and tasks"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Model aliases to evaluate (see --list)",
    )
    p.add_argument(
        "--all-models", action="store_true",
        help="Run all registered models",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Task names to evaluate (see --list)",
    )
    p.add_argument(
        "--all-tasks", action="store_true",
        help="Run all registered tasks",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List available models and tasks, then exit",
    )
    p.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    p.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    p.add_argument(
        "--val-size", type=int, default=500,
        help="Validation set size",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir", default=None,
        help="Output dir (default: results/baseline_zeroshot_<ts>)",
    )
    return p.parse_args()


def print_list():
    print("\n=== Available Models ===")
    for alias, hf_id in MODEL_REGISTRY.items():
        print(f"  {alias:<20s} -> {hf_id}")
    print("\n=== Available Tasks ===")
    for t in available_tasks():
        print(f"  {t}")
    print()


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    if not results:
        return

    tasks_seen = list(
        dict.fromkeys(r["task"] for r in results)
    )
    models_seen = list(
        dict.fromkeys(r["model_alias"] for r in results)
    )

    lookup: dict[tuple[str, str], float | None] = {}
    for r in results:
        lookup[(r["model_alias"], r["task"])] = r.get("accuracy")

    task_w = max(10, max(len(t) for t in tasks_seen))
    model_w = max(12, max(len(m) for m in models_seen))

    header = f"{'Model':<{model_w}}"
    for t in tasks_seen:
        header += f"  {t:>{task_w}}"
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("ZERO-SHOT BASELINE RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))

    for m in models_seen:
        row = f"{m:<{model_w}}"
        for t in tasks_seen:
            acc = lookup.get((m, t))
            if acc is not None:
                row += f"  {acc:>{task_w}.3f}"
            else:
                row += f"  {'ERROR':>{task_w}}"
        print(row)

    print(sep)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.list:
        print_list()
        return

    if args.all_models:
        model_aliases = list(MODEL_REGISTRY.keys())
    elif args.models:
        model_aliases = args.models
    else:
        print(
            "[error] Specify --models <alias ...> or "
            "--all-models. Use --list to see options."
        )
        return

    if args.all_tasks:
        task_names = available_tasks()
    elif args.tasks:
        task_names = args.tasks
    else:
        print(
            "[error] Specify --tasks <name ...> or "
            "--all-tasks. Use --list to see options."
        )
        return

    for alias in model_aliases:
        if alias not in MODEL_REGISTRY:
            print(
                f"[error] Unknown model alias '{alias}'. "
                "Use --list to see options."
            )
            return
    for t in task_names:
        if t not in available_tasks():
            print(
                f"[error] Unknown task '{t}'. "
                f"Available: {available_tasks()}"
            )
            return

    dtype = resolve_dtype(args.dtype, args.device)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results") / f"baseline_zeroshot_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_evals = len(model_aliases) * len(task_names)
    print(f"Output dir: {out_dir}")
    print(f"Models ({len(model_aliases)}): {model_aliases}")
    print(f"Tasks  ({len(task_names)}): {task_names}")
    print(f"Device: {args.device}  |  Dtype: {dtype}")
    print(f"Val size: {args.val_size}  |  Evals: {n_evals}")

    all_results: list[dict] = []

    for model_alias in model_aliases:
        hf_id = MODEL_REGISTRY[model_alias]
        print(f"\n{'=' * 60}")
        print(f"Loading model: {model_alias} ({hf_id})")
        print(f"{'=' * 60}")

        max_tokens_needed = max(
            _TASK_MAX_TOKENS.get(t, _DEFAULT_MAX_TOKENS)
            for t in task_names
        )

        t_load = time.perf_counter()
        try:
            backend = create_backend(
                "hf",
                model_name=hf_id,
                device=args.device,
                dtype=dtype,
                max_new_tokens=max_tokens_needed,
                do_sample=False,
            )
        except Exception as e:
            print(f"  [SKIP] Failed to load {hf_id}: {e}")
            for t in task_names:
                all_results.append({
                    "model_alias": model_alias,
                    "model_hf_id": hf_id,
                    "task": t,
                    "error": str(e),
                })
            continue

        load_time = time.perf_counter() - t_load
        print(f"  Model loaded in {load_time:.1f}s")

        for task_name in task_names:
            print(f"\n  --- Task: {task_name} ---")
            max_tok = _TASK_MAX_TOKENS.get(
                task_name, _DEFAULT_MAX_TOKENS,
            )
            backend.max_new_tokens = max_tok

            try:
                task = get_task(task_name)
                _, val_data = task.load_data(
                    train_size=0,
                    val_size=args.val_size,
                    seed=args.seed,
                )
                print(
                    f"  Val examples: {len(val_data)}"
                    f"  |  max_new_tokens: {max_tok}"
                )

                t_eval = time.perf_counter()
                eval_result = evaluate(
                    backend, val_data, task, args.batch_size,
                )
                eval_time = time.perf_counter() - t_eval

                acc = eval_result["accuracy"]
                c = eval_result["correct"]
                tot = eval_result["total"]
                pf = eval_result["parse_failures"]
                print(
                    f"  Accuracy: {acc:.4f} ({c}/{tot})"
                    f" | parse_failures: {pf}"
                    f" | time: {eval_time:.1f}s"
                )

                all_results.append({
                    "model_alias": model_alias,
                    "model_hf_id": hf_id,
                    "task": task_name,
                    "accuracy": acc,
                    "correct": c,
                    "total": tot,
                    "parse_failures": pf,
                    "eval_time_s": round(eval_time, 2),
                    "val_size": args.val_size,
                    "seed": args.seed,
                    "dtype": dtype,
                    "device": args.device,
                    "max_new_tokens": max_tok,
                })

            except Exception as e:
                print(f"  [ERROR] {task_name}: {e}")
                all_results.append({
                    "model_alias": model_alias,
                    "model_hf_id": hf_id,
                    "task": task_name,
                    "error": str(e),
                })

        del backend
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"\n  Model {model_alias} unloaded.")

    results_path = out_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    print_results_table(all_results)


if __name__ == "__main__":
    main()
