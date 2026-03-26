"""Week 10 experiment runner: top-k ablation at N=8 and pop-scaling on OPT-1.3B.

New blocks
----------
  top_k_n8       -- top-k at N=8: all_seeds / top_k=4 / top_k=2 / top_k=1
                    (top_k=1 forces --no-normalize; the z-score requires N>=2 after filtering)
  pop_scaling_1b -- population scaling N∈{1,2,4,8,16,32,64,128} on OPT-1.3B
                    at a fixed forward-pass budget (same formula as existing pop_scaling)

Both blocks run on BoolQ so results are directly comparable to existing exp_boolq results.

Usage
-----
  # top-k ablation (N=8) — run on GPU
  uv run python -m src.scripts.run_experiment_week10 --block top_k_n8 --device cuda

  # population scaling on OPT-1.3B — run on GPU
  uv run python -m src.scripts.run_experiment_week10 --block pop_scaling_1b --device cuda

  # override model (e.g. test pop_scaling_1b with a different checkpoint)
  uv run python -m src.scripts.run_experiment_week10 --block pop_scaling_1b \\
      --model facebook/opt-1.3b --device cuda

  # plot after run
  uv run python -m src.scripts.plot_results --exp-dir results/exp_top_k_n8_<ts>
  uv run python -m src.scripts.plot_results --exp-dir results/exp_pop_scaling_1b_<ts>
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared base config — calibrated HPs from existing BoolQ experiments
# ---------------------------------------------------------------------------
BASE = {
    "task":            "boolq",
    "model":           "facebook/opt-350m",
    "num_iters":       30,
    "batch_size":      16,
    "train_size":      128,
    "val_size":        277,
    "val_every":       2,
    "population_size": 8,
    "sigma":           3e-4,
    "lr":              3e-3,
    "early_stop_delta": 0,   # disable — fair budget comparison
    "device":          "cpu",
    "dtype":           "auto",
}

# Fixed forward-pass training budget (matches existing pop_scaling):
# BASE_N × BASE_iters × 2 (two-sided) × batch_size = 8 × 30 × 2 × 16 = 7680
_POP_BUDGET = BASE["population_size"] * BASE["num_iters"] * 2 * BASE["batch_size"]


def _pop_iters(n: int) -> int:
    """Iterations for population n to match _POP_BUDGET training forward passes.

    Uses ceiling so no variant falls short of the budget.
    Minimum of 2 so val_every=2 triggers at least once.
    """
    return max(2, math.ceil(_POP_BUDGET / (n * 2 * BASE["batch_size"])))


# ---------------------------------------------------------------------------
# Block definitions
# ---------------------------------------------------------------------------
BLOCKS: dict[str, dict] = {
    "top_k_n8": {
        "description": (
            "Top-k ablation at N=8: all seeds vs top_k=4 vs top_k=2 vs top_k=1. "
            "top_k=1 forces no_normalize (z-score undefined with a single sample)."
        ),
        "model": "facebook/opt-350m",
        "base_overrides": {
            "population_size": 8,
            "num_iters": _pop_iters(8),   # 30 iters — matches budget at N=8
        },
        "variants": [
            {"top_k": 0, "label": "all_seeds"},
            {"top_k": 4, "label": "top_k_4"},
            {"top_k": 2, "label": "top_k_2"},
            # top_k=1 requires normalization off (only 1 sample after filtering)
            {"top_k": 1, "no_normalize": True, "label": "top_k_1"},
        ],
    },
    "top_k_no_norm": {
        "description": (
            "top_k=1 vs top_k=2 at N=8, both with no_normalize=True — "
            "fair comparison removing the normalization confound."
        ),
        "model": "facebook/opt-350m",
        "base_overrides": {
            "population_size": 8,
            "num_iters": _pop_iters(8),
        },
        "variants": [
            {"top_k": 2, "no_normalize": True, "label": "top_k_2_nonorm"},
            {"top_k": 1, "no_normalize": True, "label": "top_k_1_nonorm"},
        ],
    },
    "pop_scaling_1b": {
        "description": (
            "Population scaling N∈{1,2,4,8,16,32,64,128} on OPT-1.3B at fixed "
            f"forward-pass budget ({_POP_BUDGET} training FPs). "
            "N=1 forces no_normalize (single-sample z-score undefined)."
        ),
        "model": "facebook/opt-1.3b",
        "base_overrides": {},
        "variants": [
            # N=1: no antithetic pair, normalization impossible with 1 sample
            {"population_size": 1,   "num_iters": _pop_iters(1),   "no_normalize": True, "label": "N1"},
            {"population_size": 2,   "num_iters": _pop_iters(2),                          "label": "N2"},
            {"population_size": 4,   "num_iters": _pop_iters(4),                          "label": "N4"},
            {"population_size": 8,   "num_iters": _pop_iters(8),                          "label": "N8"},
            {"population_size": 16,  "num_iters": _pop_iters(16),                         "label": "N16"},
            {"population_size": 32,  "num_iters": _pop_iters(32),                         "label": "N32"},
            {"population_size": 64,  "num_iters": _pop_iters(64),                         "label": "N64"},
            {"population_size": 128, "num_iters": _pop_iters(128),                        "label": "N128"},
        ],
    },
}


# ---------------------------------------------------------------------------
# CLI → train_es.py arg translation (mirrors run_experiment.py)
# ---------------------------------------------------------------------------
_BOOL_FLAGS = {"one_sided": "--one-sided", "no_normalize": "--no-normalize"}
_ARG_MAP = {
    "task":             "--task",
    "model":            "--model",
    "device":           "--device",
    "dtype":            "--dtype",
    "population_size":  "--population-size",
    "num_iters":        "--num-iters",
    "batch_size":       "--batch-size",
    "val_every":        "--val-every",
    "sigma":            "--sigma",
    "lr":               "--lr",
    "train_size":       "--train-size",
    "val_size":         "--val-size",
    "early_stop_delta": "--early-stop-delta",
    "noise_type":       "--noise-type",
    "top_k":            "--top-k",
}


def config_to_cmd(cfg: dict, out_dir: Path, seed: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.scripts.train_es",
        "--out-dir", str(out_dir),
        "--seed", str(seed),
    ]
    for key, val in cfg.items():
        if key == "label":
            continue
        if key in _BOOL_FLAGS:
            if val:
                cmd.append(_BOOL_FLAGS[key])
        elif key in _ARG_MAP:
            cmd.extend([_ARG_MAP[key], str(val)])
    return cmd


def variant_slug(variant: dict) -> str:
    if "label" in variant:
        return variant["label"]
    return "_".join(f"{k[:4]}{v}" for k, v in variant.items())


def run_variant(cfg: dict, out_dir: Path, seed: int) -> tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = config_to_cmd(cfg, out_dir, seed)
    print(f"\n  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return out_dir, result.returncode


def best_val_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "iter" and "val_acc" in entry:
            v = entry["val_acc"]
            if best is None or v > best:
                best = v
    return best


def run_block(
    block_name: str,
    model: str | None,
    n_seeds: int,
    parent_dir: Path,
    extra_base: dict,
) -> list[dict]:
    block = BLOCKS[block_name]
    # Block-level model override < CLI --model override
    block_model = model or block.get("model") or BASE["model"]
    base = {**BASE, "model": block_model, **block["base_overrides"], **extra_base}

    variants = block["variants"]
    print(f"\n{'='*60}")
    print(f"Block: {block_name} — {block['description']}")
    print(f"  {len(variants)} variant(s) × {n_seeds} seed(s) = {len(variants) * n_seeds} runs")
    print(f"  forward-pass budget per variant: {_POP_BUDGET}")
    print(f"  base: {base}")
    print(f"{'='*60}")

    results = []
    for variant in variants:
        cfg = {**base, **variant}
        slug = variant_slug(variant)
        per_variant_dir = parent_dir / slug

        best_per_seed = []
        for s in range(n_seeds):
            seed_dir = per_variant_dir / f"seed{s}"
            run_seed = 42 + s       # seeds: 42, 43, 44 — fixed, reproducible
            print(f"\n[{slug}] seed={run_seed}")
            _, rc = run_variant(cfg, seed_dir, run_seed)
            bv = best_val_from_log(seed_dir / "log.jsonl")
            best_per_seed.append(bv)
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"  -> best_val={bv}  {status}")

        valid = [v for v in best_per_seed if v is not None]
        mean_best = sum(valid) / len(valid) if valid else None
        std_best = None
        if len(valid) >= 2:
            var = sum((v - mean_best) ** 2 for v in valid) / len(valid)
            std_best = math.sqrt(var)

        results.append({
            "block": block_name,
            "variant": slug,
            "config": {k: v for k, v in variant.items() if k != "label"},
            "best_per_seed": best_per_seed,
            "mean_best_val": mean_best,
            "std_best_val": std_best,
            "run_dir": str(per_variant_dir),
        })
        std_str = f"{std_best:.4f}" if std_best is not None else "N/A"
        print(f"  [{slug}] mean_best_val={mean_best}  std={std_str}")

    return results


def print_block_summary(results: list[dict]) -> None:
    results_sorted = sorted(
        results,
        key=lambda x: x["mean_best_val"] if x["mean_best_val"] is not None else -1,
        reverse=True,
    )
    print("\n" + "=" * 60)
    print("BLOCK SUMMARY")
    print("=" * 60)
    for rank, r in enumerate(results_sorted, 1):
        mv = f"{r['mean_best_val']:.4f}" if r["mean_best_val"] is not None else "N/A"
        sv = f"{r['std_best_val']:.4f}" if r.get("std_best_val") is not None else "N/A"
        seeds_str = [f"{v:.3f}" if v is not None else "N/A" for v in r["best_per_seed"]]
        print(f"  {rank}. {r['variant']:<20} mean={mv}  std={sv}  seeds={seeds_str}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Week 10 ES experiments: top_k_n8 and pop_scaling_1b",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--block",
        required=True,
        choices=list(BLOCKS.keys()),  # top_k_n8 | top_k_no_norm | pop_scaling_1b
        help="Which experiment block to run",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Override model for all runs. Defaults: "
            "top_k_n8=facebook/opt-350m, pop_scaling_1b=facebook/opt-1.3b"
        ),
    )
    p.add_argument("--n-seeds",  type=int, default=3, help="Seeds per variant (42, 43, 44, ...)")
    p.add_argument("--out-dir",  default=None,         help="Parent output dir (default: results/exp_<block>_<ts>)")
    p.add_argument("--device",   default=None,         help="Device override, e.g. cuda or cuda:0")
    p.add_argument("--dtype",    default=None,         help="Dtype override, e.g. bfloat16")
    p.add_argument("--task",     default=None,         help="Task override (default: boolq)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    extra_base: dict = {}
    if args.device:
        extra_base["device"] = args.device
    if args.dtype:
        extra_base["dtype"] = args.dtype
    if args.task:
        extra_base["task"] = args.task

    ts = time.strftime("%Y%m%d_%H%M%S")
    parent_dir = (
        Path(args.out_dir) / args.block
        if args.out_dir
        else Path("results") / f"exp_{args.block}_{ts}"
    )

    results = run_block(args.block, args.model, args.n_seeds, parent_dir, extra_base)
    print_block_summary(results)

    parent_dir.mkdir(parents=True, exist_ok=True)
    summary_path = parent_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print(f"Plot with: uv run python -m src.scripts.plot_results --exp-dir {parent_dir}")


if __name__ == "__main__":
    main()
