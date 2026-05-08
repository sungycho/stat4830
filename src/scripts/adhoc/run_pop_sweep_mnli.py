"""Population-size scaling sweep: ES fine-tuning of Qwen2.5-1.5B-Instruct on MNLI.

Fixed LR=1e-6, fixed total budget (~15 360 train fwd passes per run).
num_iters is auto-adjusted per population size to keep budget constant:
  N=2  → 240 iters  (2×16×2×240 = 15 360)
  N=4  → 120 iters
  N=8  →  60 iters
  N=16 →  30 iters
  N=32 →  15 iters

Usage:
  uv run python -m src.scripts.adhoc.run_pop_sweep_mnli
  uv run python -m src.scripts.adhoc.run_pop_sweep_mnli --pops 4 8 16 --lr 3e-6
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
MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
TASK       = "mnli"
LR         = 1e-6
BATCH_SIZE = 16
TRAIN_SIZE = 500
VAL_SIZE   = 200
SIGMA      = 1e-3
MAX_NEW_TOKENS = 4
TARGET_FWD = 15_360   # keep constant across pop sizes

DEFAULT_POPS = [2, 4, 8, 16, 32]
# ---------------------------------------------------------------------------


def iters_for_pop(pop: int) -> int:
    """Compute num_iters so that pop × BATCH_SIZE × 2 × iters ≈ TARGET_FWD."""
    return max(1, TARGET_FWD // (pop * BATCH_SIZE * 2))


def run_one(pop: int, lr: float, out_dir: Path, reward: str, seed: int) -> dict:
    num_iters = iters_for_pop(pop)
    val_every = max(1, num_iters // 12)   # ~12 val checkpoints per run
    actual_fwd = pop * BATCH_SIZE * 2 * num_iters

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.scripts.train_es",
        "--task",            TASK,
        "--model",           MODEL,
        "--population-size", str(pop),
        "--batch-size",      str(BATCH_SIZE),
        "--num-iters",       str(num_iters),
        "--train-size",      str(TRAIN_SIZE),
        "--val-size",        str(VAL_SIZE),
        "--val-every",       str(val_every),
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
    print(f"  pop={pop}  lr={lr:.1e}  iters={num_iters}  fwd={actual_fwd}  reward={reward}")
    print(f"  out → {out_dir}")
    print(f"{'='*65}")

    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0

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
        "pop":          pop,
        "lr":           lr,
        "num_iters":    num_iters,
        "actual_fwd":   actual_fwd,
        "baseline_acc": baseline_acc,
        "best_val_acc": best_val_acc,
        "delta":        (best_val_acc - baseline_acc) if (best_val_acc is not None and baseline_acc is not None) else None,
        "elapsed_s":    round(elapsed, 1),
        "returncode":   result.returncode,
        "out_dir":      str(out_dir),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pops",   type=int,   nargs="+", default=DEFAULT_POPS,
                   help="Population sizes to sweep")
    p.add_argument("--lr",     type=float, default=LR,
                   help=f"Fixed learning rate (default: {LR:.0e})")
    p.add_argument("--reward", default="accuracy", choices=["ce", "accuracy"])
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) if args.out_dir else Path(f"results/mnli_pop_sweep_{ts}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep dir  : {sweep_dir}")
    print(f"Model      : {MODEL}")
    print(f"Task       : {TASK}")
    print(f"LR         : {args.lr:.1e}  (fixed)")
    print(f"Reward     : {args.reward}")
    print(f"Target fwd : {TARGET_FWD} per run")
    print(f"Pop sizes  : {args.pops}")
    print()
    for pop in args.pops:
        n = iters_for_pop(pop)
        print(f"  N={pop:>2d}  iters={n:>4d}  actual_fwd={pop*BATCH_SIZE*2*n}")

    summaries = []
    for pop in args.pops:
        tag = f"pop_{pop:02d}"
        out_dir = sweep_dir / tag
        summary = run_one(pop, args.lr, out_dir, args.reward, args.seed)
        summaries.append(summary)

    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Pop scaling — {TASK} × {MODEL.split('/')[-1]} — lr={args.lr:.1e}")
    print(f"{'='*65}")
    print(f"  {'N':>4s}  {'iters':>6s}  {'fwd':>7s}  {'baseline':>9s}  {'best_val':>9s}  {'delta':>8s}  {'time':>7s}")
    for s in summaries:
        b  = f"{s['baseline_acc']:.3f}" if s['baseline_acc'] is not None else "  n/a"
        bv = f"{s['best_val_acc']:.3f}" if s['best_val_acc'] is not None else "  n/a"
        d  = f"{s['delta']:+.3f}"       if s['delta']        is not None else "   n/a"
        print(f"  {s['pop']:>4d}  {s['num_iters']:>6d}  {s['actual_fwd']:>7d}  {b:>9s}  {bv:>9s}  {d:>8s}  {s['elapsed_s']:>6.0f}s")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
