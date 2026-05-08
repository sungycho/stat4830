"""Analyze predicted answer distribution for specific model×task pairs.

Runs the model on a sample of examples, collects the predicted label for each
(via task.predict()), and reports what fraction of outputs fall into each
label category vs. gold. Useful for diagnosing yes-man / constant-output failure.

Usage:
  uv run python -m src.scripts.adhoc.analyze_answer_dist
  uv run python -m src.scripts.adhoc.analyze_answer_dist --n 200
  uv run python -m src.scripts.adhoc.analyze_answer_dist --cases opt-350m:wic opt-2.7b:wsc
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task, resolve_prompt_config

# ---------------------------------------------------------------------------
# Model registry (short-name → HF id, instruct flag)
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "opt-350m":              ("facebook/opt-350m",            False),
    "opt-1.3b":              ("facebook/opt-1.3b",            False),
    "opt-2.7b":              ("facebook/opt-2.7b",            False),
    "opt-13b":               ("facebook/opt-13b",             False),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct",  True),
    "qwen2.5-3b-instruct":   ("Qwen/Qwen2.5-3B-Instruct",    True),
    "qwen2.5-7b-instruct":   ("Qwen/Qwen2.5-7B-Instruct",    True),
    "qwen2.5-math-1.5b":     ("Qwen/Qwen2.5-Math-1.5B",      False),
    "qwen2.5-math-7b":       ("Qwen/Qwen2.5-Math-7B",        False),
}

# ---------------------------------------------------------------------------
# Default cases to analyze  (moderate-ρ, low-p₀ bucket)
# ---------------------------------------------------------------------------
DEFAULT_CASES = [
    ("opt-1.3b",              "drop"),
    ("opt-1.3b",              "mnli"),
    ("opt-1.3b",              "record"),
    ("opt-2.7b",              "drop"),
    ("opt-2.7b",              "record"),
    ("opt-350m",              "cb"),
    ("opt-350m",              "drop"),
    ("opt-350m",              "record"),
    ("qwen2.5-1.5b-instruct", "mnli"),
]

MAX_NEW_TOKENS = {
    "copa":   8,
    "wsc":    4,
    "wic":    4,
    "cb":     4,
    "mnli":   4,
    "drop":   8,
    "record": 16,
}
# ---------------------------------------------------------------------------


def _resolve_model(model_str: str) -> tuple[str, bool]:
    """Return (hf_name, is_instruct). Accepts short names or full HF paths."""
    if model_str in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_str]
    # full HF path passed directly — infer instruct flag from name
    is_instruct = "instruct" in model_str.lower()
    return model_str, is_instruct


def analyze(model_hf: str, task_name: str, n: int, prompt_style: str, no_chat: bool) -> dict:
    print(f"\n{'='*65}")
    print(f"  Model : {model_hf}")
    print(f"  Task  : {task_name}  (n={n})")
    print(f"{'='*65}")

    task = get_task(task_name)
    _, val_data = task.load_data(train_size=n, val_size=n, seed=42)
    examples = val_data[:n]

    prompt_cfg = resolve_prompt_config(task, prompt_style)
    raw = no_chat or prompt_cfg.force_raw or task.prefer_base_prompt

    backend = create_backend(
        "hf",
        model_name=model_hf,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        max_new_tokens=MAX_NEW_TOKENS.get(task_name, 8),
        do_sample=False,
    )

    prompts = [prompt_cfg.prompt_fn(ex) for ex in examples]
    outputs = backend.generate_batch(prompts, raw=raw)

    pred_counts  = Counter()
    gold_counts  = Counter()
    score_counts = Counter()

    for out, ex in zip(outputs, examples):
        pred = task.predict(out) if hasattr(task, "predict") else _fallback_predict(out)
        pred_counts[pred if pred is not None else "<parse_fail>"] += 1
        gold_counts[task.gold_label(ex)] += 1
        sc = task.score(out, ex)
        score_counts[">0 (correct)" if sc > 0 else ("=0 (parse fail)" if sc == 0.0 else "<0 (wrong)")] += 1

    total = len(examples)

    print(f"\n  Predicted label distribution (n={total}):")
    for label, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total * 40)
        print(f"    {str(label):>14s}  {cnt:4d} / {total}  ({cnt/total*100:5.1f}%)  {bar}")

    print(f"\n  Gold label distribution:")
    for label, cnt in sorted(gold_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total * 40)
        print(f"    {str(label):>14s}  {cnt:4d} / {total}  ({cnt/total*100:5.1f}%)  {bar}")

    print(f"\n  Score breakdown:")
    for label, cnt in sorted(score_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:>20s}  {cnt:4d} / {total}  ({cnt/total*100:5.1f}%)")

    # Yes-man diagnosis
    top_pred, top_cnt = pred_counts.most_common(1)[0]
    if top_pred != "<parse_fail>" and top_cnt / total > 0.8:
        diagnosis = f"YES-MAN: '{top_pred}' in {top_cnt/total*100:.1f}% of outputs"
        print(f"\n  ⚠  {diagnosis}")
    elif pred_counts.get("<parse_fail>", 0) / total > 0.8:
        diagnosis = f"PARSE FAILURE: {pred_counts['<parse_fail>']/total*100:.1f}% unparseable"
        print(f"\n  ⚠  {diagnosis}")
    else:
        diagnosis = f"no dominance (top='{top_pred}' {top_cnt/total*100:.1f}%)"
        print(f"\n  ✓  {diagnosis}")

    del backend
    torch.cuda.empty_cache()

    return {
        "model":             model_hf,
        "task":              task_name,
        "n":                 total,
        "prompt_style":      prompt_style,
        "pred_distribution": dict(pred_counts),
        "gold_distribution": dict(gold_counts),
        "score_breakdown":   dict(score_counts),
        "diagnosis":         diagnosis,
    }


def _fallback_predict(text: str) -> str | None:
    return text.strip()[:20] if text.strip() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases",  nargs="*", default=None,
                   help="model:task pairs, e.g. opt-350m:wic or facebook/opt-350m:wic")
    p.add_argument("--n",      type=int, default=200,
                   help="Number of val examples to evaluate (default: 200)")
    p.add_argument("--prompt-style", default=None,
                   choices=["simple", "complex", "mezo", "free"],
                   help="Override prompt style for all cases (default: auto per model)")
    p.add_argument("--no-chat-template", action="store_true", default=False,
                   help="Force raw generation (no chat template) for all cases")
    p.add_argument("--output", default=None,
                   help="Path to save JSON results (default: results/answer_dist_<timestamp>.json)")
    args = p.parse_args()

    if args.cases:
        raw_cases = []
        for c in args.cases:
            model, _, task = c.rpartition(":")
            raw_cases.append((model, task))
    else:
        raw_cases = DEFAULT_CASES

    all_results = []
    for model_str, task_name in raw_cases:
        hf_name, is_instruct = _resolve_model(model_str)
        # Per-case style: instruct → simple+chat; base → complex+no-chat
        if args.prompt_style is not None:
            style = args.prompt_style
            no_chat = args.no_chat_template
        else:
            style   = "simple" if is_instruct else "complex"
            no_chat = not is_instruct
        result = analyze(hf_name, task_name, args.n,
                         prompt_style=style,
                         no_chat=no_chat)
        all_results.append(result)

    out_path = Path(args.output) if args.output else Path(
        f"results/answer_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
