"""ES fine-tuning at N=2 with normalization on vs off, for any (model, task).

Mirrors the pattern of `gsm8k_n2_norm_{on,off}_s4*` runs but parameterized.
Auto-detects base vs instruct models (registry below) and per-task
max_new_tokens. Two runs per seed (norm on, norm off); summary written to
`<out_dir>/sweep_summary.json`.

Usage:
  uv run python -m src.scripts.adhoc.run_norm_n2 --model llama3.2-1b --task math500
  uv run python -m src.scripts.adhoc.run_norm_n2 --model llama3.2-1b --task math500 --seeds 42 43 44
  uv run python -m src.scripts.adhoc.run_norm_n2 --model qwen2.5-1.5b-instruct --task gsm8k --lr-on 1e-4 --lr-off 1e-6
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
# Registries (kept in sync with run_rho_sweep.py)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple[str, bool]] = {
    # name → (hf_id, is_instruct)
    "opt-350m":              ("facebook/opt-350m",            False),
    "opt-1.3b":              ("facebook/opt-1.3b",            False),
    "opt-2.7b":              ("facebook/opt-2.7b",            False),
    "opt-13b":               ("facebook/opt-13b",             False),
    "llama2-7b":             ("meta-llama/Llama-2-7b-hf",     False),
    "llama2-13b":            ("meta-llama/Llama-2-13b-hf",    False),
    "llama3-8b":             ("meta-llama/Meta-Llama-3-8B",   False),
    "llama3.1-8b":           ("meta-llama/Meta-Llama-3.1-8B", False),
    "llama3.2-1b":           ("meta-llama/Llama-3.2-1B",      False),
    "llama3.2-3b":           ("meta-llama/Llama-3.2-3B",      False),
    "qwen2.5-math-1.5b":     ("Qwen/Qwen2.5-Math-1.5B",       False),
    "qwen2.5-math-7b":       ("Qwen/Qwen2.5-Math-7B",         False),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct",   True),
    "qwen2.5-3b-instruct":   ("Qwen/Qwen2.5-3B-Instruct",     True),
    "qwen2.5-7b-instruct":   ("Qwen/Qwen2.5-7B-Instruct",     True),
}

TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "sst2": 4, "sst5": 4, "rte": 4, "boolq": 4, "mnli": 4, "cb": 4,
    "wsc": 4, "wic": 4, "copa": 8, "trec": 4,
    "squad": 32, "drop": 8, "record": 16,
    "gsm8k": 256, "math500": 256, "countdown": 256,
}

# Tasks that have label_words() and therefore support --reward ce
CE_REWARD_TASKS = {"sst2", "sst5", "rte", "boolq", "mnli", "cb", "wsc", "wic", "trec"}

# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------
POP_SIZE   = 2
BATCH_SIZE = 16
NUM_ITERS  = 100
TRAIN_SIZE = 128
VAL_SIZE   = 200
VAL_EVERY  = 5
SIGMA      = 1e-3
LR_ON      = 1e-4   
LR_OFF     = 1e-4
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> tuple[str, bool]:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # Allow passing full HF id directly; assume instruct if name contains "instruct"
    return name, "instruct" in name.lower() or "chat" in name.lower()


def run_one(
    *, hf_model: str, instruct: bool, task: str, normalize: bool, lr: float,
    sigma: float, pop: int, batch: int, num_iters: int, train_size: int,
    val_size: int, val_every: int, max_new_tokens: int, reward: str,
    seed: int, out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.scripts.train_es",
        "--task",            task,
        "--model",           hf_model,
        "--population-size", str(pop),
        "--batch-size",      str(batch),
        "--num-iters",       str(num_iters),
        "--train-size",      str(train_size),
        "--val-size",        str(val_size),
        "--val-every",       str(val_every),
        "--sigma",           str(sigma),
        "--lr",              str(lr),
        "--max-new-tokens",  str(max_new_tokens),
        "--reward",          reward,
        "--seed",            str(seed),
        "--out-dir",         str(out_dir),
        "--no-save",
    ]
    if instruct:
        cmd += ["--prompt-style", "simple", "--chat-template"]
    else:
        cmd += ["--prompt-style", "complex", "--no-chat-template"]
    if not normalize:
        cmd += ["--no-normalize"]

    label = "norm_on" if normalize else "norm_off"
    print(f"\n{'='*70}")
    print(f"  {label}  seed={seed}  lr={lr:.1e}  sigma={sigma:.1e}")
    print(f"  out → {out_dir}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    elapsed = time.perf_counter() - t0

    log_path = out_dir / "log.jsonl"
    baseline_acc = best_val_acc = None
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
        "label":        label,
        "seed":         seed,
        "lr":           lr,
        "normalize":    normalize,
        "baseline_acc": baseline_acc,
        "best_val_acc": best_val_acc,
        "delta":        (best_val_acc - baseline_acc) if (best_val_acc is not None and baseline_acc is not None) else None,
        "elapsed_s":    round(elapsed, 1),
        "returncode":   rc,
        "out_dir":      str(out_dir),
    }


def main():
    p = argparse.ArgumentParser(description="N=2 ES fine-tune: normalization on vs off")
    p.add_argument("--model", required=True,
                   help=f"Model name (registry key or full HF id). Registry: {sorted(MODEL_REGISTRY)}")
    p.add_argument("--task", required=True,
                   help=f"Task name (must exist in src/tasks/). Known: {sorted(TASK_MAX_NEW_TOKENS)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds to run (default: just 42)")
    p.add_argument("--lr-on",  type=float, default=LR_ON,  help=f"LR for norm-on runs (default {LR_ON:.0e})")
    p.add_argument("--lr-off", type=float, default=LR_OFF, help=f"LR for norm-off runs (default {LR_OFF:.0e})")
    p.add_argument("--sigma",  type=float, default=SIGMA)
    p.add_argument("--pop",    type=int,   default=POP_SIZE)
    p.add_argument("--batch-size",      type=int, default=BATCH_SIZE)
    p.add_argument("--num-iters",       type=int, default=NUM_ITERS)
    p.add_argument("--train-size",      type=int, default=TRAIN_SIZE)
    p.add_argument("--val-size",        type=int, default=VAL_SIZE)
    p.add_argument("--val-every",       type=int, default=VAL_EVERY)
    p.add_argument("--max-new-tokens",  type=int, default=None,
                   help="Override per-task default")
    p.add_argument("--reward", default=None, choices=["accuracy", "ce"],
                   help="Default: 'ce' for classification tasks, 'accuracy' otherwise")
    p.add_argument("--out-dir", default=None,
                   help="Parent dir (default: results/norm_n2_<model>_<task>_<ts>)")
    p.add_argument("--only", choices=["on", "off"], default=None,
                   help="Run only norm-on or norm-off (default: both)")
    args = p.parse_args()

    hf_model, instruct = resolve_model(args.model)
    if args.task not in TASK_MAX_NEW_TOKENS and args.max_new_tokens is None:
        raise SystemExit(f"Unknown task '{args.task}' — pass --max-new-tokens explicitly")
    max_new_tokens = args.max_new_tokens or TASK_MAX_NEW_TOKENS[args.task]
    reward = args.reward or ("ce" if args.task in CE_REWARD_TASKS else "accuracy")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(args.out_dir) if args.out_dir else Path(
        f"results/norm_n2_{args.model.replace('/', '-')}_{args.task}_{ts}"
    )
    parent.mkdir(parents=True, exist_ok=True)

    fwd_per_run = args.pop * args.batch_size * 2 * args.num_iters
    print(f"Out dir       : {parent}")
    print(f"Model         : {args.model}  →  {hf_model}  (instruct={instruct})")
    print(f"Task          : {args.task}  (max_new_tokens={max_new_tokens}, reward={reward})")
    print(f"Pop / batch   : {args.pop} / {args.batch_size}")
    print(f"Iters / size  : {args.num_iters} iters, train={args.train_size}, val={args.val_size}")
    print(f"Sigma         : {args.sigma:.1e}")
    print(f"LR on / off   : {args.lr_on:.1e} / {args.lr_off:.1e}")
    print(f"Seeds         : {args.seeds}")
    print(f"Budget/run    : {fwd_per_run} train fwd passes")
    runs_planned = (1 if args.only else 2) * len(args.seeds)
    print(f"Total runs    : {runs_planned}")

    summaries = []
    for seed in args.seeds:
        if args.only != "off":
            tag = f"norm_on_s{seed}"
            summaries.append(run_one(
                hf_model=hf_model, instruct=instruct, task=args.task, normalize=True,
                lr=args.lr_on, sigma=args.sigma, pop=args.pop, batch=args.batch_size,
                num_iters=args.num_iters, train_size=args.train_size, val_size=args.val_size,
                val_every=args.val_every, max_new_tokens=max_new_tokens, reward=reward,
                seed=seed, out_dir=parent / tag,
            ))
        if args.only != "on":
            tag = f"norm_off_s{seed}"
            summaries.append(run_one(
                hf_model=hf_model, instruct=instruct, task=args.task, normalize=False,
                lr=args.lr_off, sigma=args.sigma, pop=args.pop, batch=args.batch_size,
                num_iters=args.num_iters, train_size=args.train_size, val_size=args.val_size,
                val_every=args.val_every, max_new_tokens=max_new_tokens, reward=reward,
                seed=seed, out_dir=parent / tag,
            ))

    summary_path = parent / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Norm on/off — N={args.pop} — {args.task} × {args.model}")
    print(f"{'='*70}")
    print(f"  {'label':>9s}  {'seed':>4s}  {'lr':>9s}  {'baseline':>9s}  {'best_val':>9s}  {'delta':>8s}  {'time':>7s}  rc")
    for s in summaries:
        b  = f"{s['baseline_acc']:.3f}" if s['baseline_acc'] is not None else "  n/a"
        bv = f"{s['best_val_acc']:.3f}" if s['best_val_acc'] is not None else "  n/a"
        d  = f"{s['delta']:+.3f}"       if s['delta']        is not None else "   n/a"
        print(f"  {s['label']:>9s}  {s['seed']:>4d}  {s['lr']:>9.1e}  {b:>9s}  {bv:>9s}  {d:>8s}  {s['elapsed_s']:>6.0f}s  {s['returncode']}")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
