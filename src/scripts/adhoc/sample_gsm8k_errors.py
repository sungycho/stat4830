"""Sample GSM8K outputs from Qwen2.5-1.5B-Instruct until we have N correct and N wrong.

Runs examples one at a time (greedy, no sampling) and logs:
  - question
  - gold answer (number + full answer string)
  - model raw output
  - predicted number extracted from output
  - correct / wrong
  - failure mode tag (parse_fail, wrong_number, correct)

Stops once we have `--n-each` correct AND `--n-each` wrong examples.
Saves everything to a JSON log.

Usage:
  uv run python -m src.scripts.sample_gsm8k_errors \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --n-each 5 \\
      --max-examples 100 \\
      --max-new-tokens 512 \\
      --output results/gsm8k_error_samples.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

_FINAL_NUMBER = re.compile(r"####\s*([\d,]+(?:\.\d+)?)")
_NUMBERS = re.compile(r"-?[\d,]+(?:\.\d+)?")


def parse_gold(answer_str: str) -> float | None:
    m = _FINAL_NUMBER.search(answer_str)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def parse_pred(text: str) -> float | None:
    # prefer #### pattern first (model sometimes follows the format)
    m = _FINAL_NUMBER.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # fall back to last number in output
    matches = _NUMBERS.findall(text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def failure_tag(pred: float | None, gold: float) -> str:
    if pred is None:
        return "parse_fail"
    if abs(pred - gold) < 1e-3:
        return "correct"
    return "wrong_number"


def build_prompt(example: dict) -> str:
    return (
        f'{example["question"]}\n'
        "Solve step by step. End with the final numeric answer after ####."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-each",         type=int, default=5,
                   help="Number of correct and wrong examples to collect")
    p.add_argument("--max-examples",   type=int, default=100,
                   help="Give up after this many examples if targets not reached")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--dtype",          default="bfloat16")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output",         default="results/gsm8k_error_samples.json")
    args = p.parse_args()

    # lazy import so the module loads fast even without torch
    from src.backends.hf_backend import HFBackend

    print(f"[gsm8k-sampler] Loading model: {args.model}")
    backend = HFBackend(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    print("[gsm8k-sampler] Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main")["test"]
    ds = ds.shuffle(seed=args.seed)

    correct_samples: list[dict] = []
    wrong_samples:   list[dict] = []
    seen = 0

    for ex in ds:
        if seen >= args.max_examples:
            print(f"[gsm8k-sampler] Reached max_examples={args.max_examples}, stopping.")
            break
        if len(correct_samples) >= args.n_each and len(wrong_samples) >= args.n_each:
            break

        gold_num = parse_gold(ex["answer"])
        if gold_num is None:
            continue  # skip malformed examples

        prompt = build_prompt(ex)
        raw_output = backend.generate(prompt)
        pred_num = parse_pred(raw_output)
        tag = failure_tag(pred_num, gold_num)
        seen += 1

        record = {
            "question":     ex["question"],
            "gold_answer":  ex["answer"],
            "gold_number":  gold_num,
            "model_output": raw_output,
            "pred_number":  pred_num,
            "tag":          tag,
        }

        if tag == "correct" and len(correct_samples) < args.n_each:
            correct_samples.append(record)
            print(f"  [correct {len(correct_samples)}/{args.n_each}] pred={pred_num}  gold={gold_num}")
        elif tag != "correct" and len(wrong_samples) < args.n_each:
            wrong_samples.append(record)
            print(f"  [wrong   {len(wrong_samples)}/{args.n_each}] tag={tag}  pred={pred_num}  gold={gold_num}")
        else:
            # already have enough of this category; keep looping for the other
            print(f"  [skip {tag}] already have enough — continuing for the other category")

        print(f"     progress: correct={len(correct_samples)}/{args.n_each}  wrong={len(wrong_samples)}/{args.n_each}  seen={seen}")

    out = {
        "model":           args.model,
        "n_each":          args.n_each,
        "total_seen":      seen,
        "correct_samples": correct_samples,
        "wrong_samples":   wrong_samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[gsm8k-sampler] Done. {len(correct_samples)} correct, {len(wrong_samples)} wrong.")
    print(f"[gsm8k-sampler] Saved to: {args.output}")

    # Pretty-print summary to stdout
    _print_summary(out)


def _print_summary(out: dict) -> None:
    sep = "=" * 72

    def _show(samples: list[dict], label: str) -> None:
        print(f"\n{sep}")
        print(f"  {label} ({len(samples)} examples)")
        print(sep)
        for i, s in enumerate(samples, 1):
            print(f"\n--- Example {i} ---")
            print(f"QUESTION:\n{s['question']}")
            print(f"\nGOLD ANSWER (number): {s['gold_number']}")
            print(f"GOLD FULL ANSWER:\n{s['gold_answer']}")
            print(f"\nMODEL OUTPUT:\n{s['model_output']}")
            print(f"\nPREDICTED NUMBER: {s['pred_number']}")
            print(f"TAG: {s['tag']}")

    _show(out["correct_samples"], "CORRECT EXAMPLES")
    _show(out["wrong_samples"],   "WRONG EXAMPLES")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
