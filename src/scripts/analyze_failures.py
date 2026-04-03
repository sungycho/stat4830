"""Failure mode analysis for BoolQ base model.

Categorises each val example as:
  - correct      : model predicted the right label
  - wrong_answer : model output contained yes/no but predicted wrong label
  - format_error : model output contained neither yes nor no

Also reports class balance and sample raw outputs for each failure type.

Usage:
  uv run python -m src.scripts.analyze_failures
  uv run python -m src.scripts.analyze_failures --model facebook/opt-1.3b --val-size 500
  uv run python -m src.scripts.analyze_failures --checkpoint results/exp_week9/.../best.pt
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import textwrap
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task
from src.utils.seeds import set_seeds


_YES_RE = __import__("re").compile(r"\byes\b", __import__("re").IGNORECASE)
_NO_RE  = __import__("re").compile(r"\bno\b",  __import__("re").IGNORECASE)


def classify_output(text: str, example: dict) -> str:
    """Return 'correct', 'wrong_answer', or 'format_error'."""
    yes = _YES_RE.search(text)
    no  = _NO_RE.search(text)
    if yes and no:
        pred = 1 if yes.start() < no.start() else 0
    elif yes:
        pred = 1
    elif no:
        pred = 0
    else:
        return "format_error"
    return "correct" if pred == example["label"] else "wrong_answer"


def parse_args():
    p = argparse.ArgumentParser(
        description="BoolQ failure-mode analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       default=os.environ.get("MODEL_NAME", "facebook/opt-350m"))
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",       default="bfloat16")
    p.add_argument("--val-size",    type=int, default=500)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="Slightly more than 4 so we can see what the model actually writes")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Optional .pt checkpoint to load (e.g. best.pt from a run)")
    p.add_argument("--n-examples",  type=int, default=5,
                   help="Number of raw examples to print per failure category")
    p.add_argument("--out",         type=str, default=None,
                   help="Optional path to save full results as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    backend = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        backend.model.load_state_dict(ckpt)
        print(f"[analyze] loaded checkpoint: {args.checkpoint}")

    task = get_task("boolq")
    _, val_data = task.load_data(train_size=1, val_size=args.val_size, seed=args.seed)

    # ── class balance ────────────────────────────────────────────────────────
    label_counts = collections.Counter(ex["label"] for ex in val_data)
    n_yes = label_counts[1]
    n_no  = label_counts[0]
    total = len(val_data)
    majority_label = "yes" if n_yes >= n_no else "no"
    majority_acc   = max(n_yes, n_no) / total

    print(f"\n{'='*60}")
    print(f"CLASS BALANCE  (n={total})")
    print(f"  yes (label=1): {n_yes:4d}  ({100*n_yes/total:.1f}%)")
    print(f"  no  (label=0): {n_no:4d}  ({100*n_no/total:.1f}%)")
    print(f"  majority-class baseline (always '{majority_label}'): {majority_acc:.3f}")
    print(f"  random-guess baseline:  0.500")
    print(f"{'='*60}\n")

    # ── run inference ────────────────────────────────────────────────────────
    results = []  # list of dicts: {category, raw_output, label, question, passage_snippet}
    correct = wrong_answer = format_error = 0

    for i in range(0, total, args.batch_size):
        chunk  = val_data[i : i + args.batch_size]
        prompts = [task.build_prompt(ex) for ex in chunk]
        outputs = backend.generate_batch(prompts)

        for ex, out in zip(chunk, outputs):
            cat = classify_output(out, ex)
            if cat == "correct":
                correct += 1
            elif cat == "wrong_answer":
                wrong_answer += 1
            else:
                format_error += 1

            results.append({
                "category":        cat,
                "raw_output":      out,
                "label":           ex["label"],   # 0=no, 1=yes
                "question":        ex["question"],
                "passage_snippet": ex["passage"][:120],
            })

        done = min(i + args.batch_size, total)
        print(f"  [{done}/{total}]  correct={correct}  wrong={wrong_answer}  format_err={format_error}",
              end="\r")

    print()  # newline after progress

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS  (n={total})")
    print(f"  correct      : {correct:4d}  ({100*correct/total:.1f}%)")
    print(f"  wrong_answer : {wrong_answer:4d}  ({100*wrong_answer/total:.1f}%)")
    print(f"  format_error : {format_error:4d}  ({100*format_error/total:.1f}%)")
    print(f"  val_acc (task scoring): {correct/total:.3f}")
    print(f"{'='*60}")

    # ── baseline comparisons ─────────────────────────────────────────────────
    print(f"\nBASELINE COMPARISONS")
    print(f"  model val_acc        : {correct/total:.3f}")
    print(f"  majority-class acc   : {majority_acc:.3f}  (always '{majority_label}')")
    print(f"  random-guess acc     : 0.500")
    if correct / total < majority_acc:
        gap = majority_acc - correct / total
        print(f"  !! model is {gap:.3f} BELOW majority-class baseline !!")

    # ── format error breakdown by true label ────────────────────────────────
    fmt_yes = sum(1 for r in results if r["category"] == "format_error" and r["label"] == 1)
    fmt_no  = sum(1 for r in results if r["category"] == "format_error" and r["label"] == 0)
    if format_error > 0:
        print(f"\nFORMAT ERRORS by true label:")
        print(f"  true=yes: {fmt_yes}  ({100*fmt_yes/format_error:.1f}% of format errors)")
        print(f"  true=no : {fmt_no}  ({100*fmt_no/format_error:.1f}% of format errors)")

    # ── wrong answer breakdown by true label ────────────────────────────────
    wa_yes = sum(1 for r in results if r["category"] == "wrong_answer" and r["label"] == 1)
    wa_no  = sum(1 for r in results if r["category"] == "wrong_answer" and r["label"] == 0)
    if wrong_answer > 0:
        print(f"\nWRONG ANSWERS by true label:")
        print(f"  true=yes (predicted no) : {wa_yes}  ({100*wa_yes/wrong_answer:.1f}% of wrong answers)")
        print(f"  true=no  (predicted yes): {wa_no}  ({100*wa_no/wrong_answer:.1f}% of wrong answers)")

    # ── raw output samples ───────────────────────────────────────────────────
    for cat in ("format_error", "wrong_answer", "correct"):
        samples = [r for r in results if r["category"] == cat][: args.n_examples]
        if not samples:
            continue
        label_str = {0: "no", 1: "yes"}
        print(f"\n{'─'*60}")
        print(f"SAMPLES: {cat.upper()}  (showing {len(samples)} of {sum(1 for r in results if r['category']==cat)})")
        for j, r in enumerate(samples, 1):
            print(f"\n  [{j}] true_label={label_str[r['label']]}")
            print(f"      question : {r['question']}")
            print(f"      passage  : {textwrap.shorten(r['passage_snippet'], 80)}")
            print(f"      output   : {repr(r['raw_output'])}")

    # ── save full results ────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "val_size": total,
            "class_balance": {"yes": n_yes, "no": n_no},
            "majority_acc": majority_acc,
            "summary": {
                "correct": correct,
                "wrong_answer": wrong_answer,
                "format_error": format_error,
                "val_acc": correct / total,
            },
            "examples": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n[analyze] saved full results → {out_path}")


if __name__ == "__main__":
    main()
