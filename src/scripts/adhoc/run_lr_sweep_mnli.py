"""LR sweep: ES fine-tuning of Qwen2.5-1.5B-Instruct on MNLI.

Fixed budget: ~5 120 train fwd passes per run
  (pop=8 × batch=16 × 2 antithetic × 20 iters = 5 120)

Defaults to --reward ce (restricted log-softmax over label words) to avoid
the 87% parse-fail rate observed at baseline for this model×task pair.

Usage:
  uv run python -m src.scripts.adhoc.run_lr_sweep_mnli
  uv run python -m src.scripts.adhoc.run_lr_sweep_mnli --reward accuracy
  uv run python -m src.scripts.adhoc.run_lr_sweep_mnli --lrs 1e-5 1e-6 1e-7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------
MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
TASK       = "mnli"
POP_SIZE   = 8
BATCH_SIZE = 16
NUM_ITERS  = 20     # → 8×16×2×20 = 5 120 train fwd passes
TRAIN_SIZE = 500
VAL_SIZE   = 200
VAL_EVERY  = 5
SIGMA      = 1e-3
MAX_NEW_TOKENS = 4  # only matters when --reward accuracy

DEFAULT_LRS = [1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
# ---------------------------------------------------------------------------


def run_one(lr: float, out_dir: Path, reward: str, seed: int, num_iters: int = NUM_ITERS) -> dict:
    """Launch a single train_es run and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.scripts.train_es",
        "--task",            TASK,
        "--model",           MODEL,
        "--population-size", str(POP_SIZE),
        "--batch-size",      str(BATCH_SIZE),
        "--num-iters",       str(num_iters),
        "--train-size",      str(TRAIN_SIZE),
        "--val-size",        str(VAL_SIZE),
        "--val-every",       str(VAL_EVERY),
        "--sigma",           str(SIGMA),
        "--lr",              str(lr),
        "--max-new-tokens",  str(MAX_NEW_TOKENS),
        "--prompt-style",    "simple",
        "--chat-template",
        "--reward",          reward,
        "--seed",            str(seed),
        "--out-dir",         str(out_dir),
        "--no-save",
    ]

    t0 = time.perf_counter()
    print(f"\n{'='*65}")
    print(f"  LR={lr:.1e}  reward={reward}  seed={seed}")
    print(f"  out → {out_dir}")
    print(f"{'='*65}")

    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0

    # Parse best val_acc from log.jsonl
    log_path = out_dir / "log.jsonl"
    best_val_acc = None
    baseline_acc = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "baseline":
                baseline_acc = entry.get("val_acc")
            if entry.get("event") == "iter" and "val_acc" in entry:
                v = entry["val_acc"]
                if best_val_acc is None or v > best_val_acc:
                    best_val_acc = v

    return {
        "lr":           lr,
        "baseline_acc": baseline_acc,
        "best_val_acc": best_val_acc,
        "delta":        (best_val_acc - baseline_acc) if (best_val_acc is not None and baseline_acc is not None) else None,
        "elapsed_s":    round(elapsed, 1),
        "returncode":   result.returncode,
        "out_dir":      str(out_dir),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs",       type=float, nargs="+", default=DEFAULT_LRS,
                   help="Learning rates to sweep")
    p.add_argument("--num-iters", type=int, default=NUM_ITERS,
                   help=f"ES iterations per run (default: {NUM_ITERS}; fwd = pop×batch×2×iters)")
    p.add_argument("--reward", default="ce", choices=["ce", "accuracy"],
                   help="Reward type: 'ce' (default) avoids parse-fail; 'accuracy' uses binary generation reward")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--out-dir", default=None,
                   help="Parent output dir (default: results/mnli_lr_sweep_<timestamp>)")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) if args.out_dir else Path(f"results/mnli_lr_sweep_{ts}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    fwd_per_run = POP_SIZE * BATCH_SIZE * 2 * args.num_iters
    print(f"Sweep dir    : {sweep_dir}")
    print(f"Model        : {MODEL}")
    print(f"Task         : {TASK}")
    print(f"LRs          : {args.lrs}")
    print(f"Reward       : {args.reward}")
    print(f"Budget/run   : {fwd_per_run} train fwd passes ({args.num_iters} iters × pop{POP_SIZE} × batch{BATCH_SIZE} × 2)")
    print(f"Total budget : {fwd_per_run * len(args.lrs)} fwd passes")

    summaries = []
    for lr in args.lrs:
        lr_tag = f"lr_{lr:.0e}".replace("+", "")
        out_dir = sweep_dir / lr_tag
        summary = run_one(lr, out_dir, args.reward, args.seed, args.num_iters)
        summaries.append(summary)

    # Save sweep summary
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Print results table
    print(f"\n{'='*65}")
    print(f"  Sweep complete — {TASK} × {MODEL.split('/')[-1]} — reward={args.reward}")
    print(f"{'='*65}")
    print(f"  {'LR':>10s}  {'baseline':>9s}  {'best_val':>9s}  {'delta':>8s}  {'time':>7s}  rc")
    for s in summaries:
        b  = f"{s['baseline_acc']:.3f}" if s['baseline_acc'] is not None else "   n/a"
        bv = f"{s['best_val_acc']:.3f}" if s['best_val_acc'] is not None else "   n/a"
        d  = f"{s['delta']:+.3f}"       if s['delta']        is not None else "    n/a"
        print(f"  {s['lr']:>10.1e}  {b:>9s}  {bv:>9s}  {d:>8s}  {s['elapsed_s']:>6.0f}s  {s['returncode']}")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
