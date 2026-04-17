"""Structured experiment runner for the ES variant comparison study.

Experiment blocks (matching the game plan):
  calibration   -- sigma/LR grid on RTE (required first to pick best HP)
  one_vs_two    -- one-sided vs two-sided ES
  noise_type    -- Gaussian vs Rademacher noise
  normalize     -- normalised vs unnormalised reward update
  top_k         -- ES vs ARS-style top-k selection
  pop_scaling   -- population size N ∈ {4, 8, 16, 32}

Each variant is run with --n-seeds independent seeds.

Usage:
  uv run python -m src.scripts.run_experiment --block calibration
  uv run python -m src.scripts.run_experiment --block all --model facebook/opt-350m
  uv run python -m src.scripts.run_experiment --block one_vs_two --n-seeds 3

Results land in: results/exp_<block>_<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared base config — override per-block as needed
# ---------------------------------------------------------------------------
BASE = {
    "task": "rte",
    "model": "facebook/opt-350m",
    "num_iters": 30,
    "batch_size": 16,
    "train_size": 128,
    "val_size": 277,   # full RTE validation set
    "val_every": 2,
    "population_size": 8,
    "sigma": 3e-4,
    "lr": 3e-3,
    "early_stop_delta": 0,  # disable for fair budget comparison
    "device": "cpu",
    "dtype": "auto",
}

# Fixed forward-pass training budget for pop_scaling.
# = BASE population_size × BASE num_iters × 2 (two-sided) × batch_size
# = 8 × 30 × 2 × 16 = 7680 training forward passes.
_POP_BUDGET = BASE["population_size"] * BASE["num_iters"] * 2 * BASE["batch_size"]


def _pop_iters(n: int) -> int:
    """Iters needed for population size n to consume _POP_BUDGET training fwd passes.

    Uses ceiling so every variant meets or slightly exceeds the budget
    (never falls short due to truncation).
    """
    return max(2, math.ceil(_POP_BUDGET / (n * 2 * BASE["batch_size"])))


# ---------------------------------------------------------------------------
# Block definitions — each entry in "variants" is a dict of CLI overrides
# ---------------------------------------------------------------------------
BLOCKS: dict[str, dict] = {
    "calibration": {
        "description": "Sigma/LR calibration sweep on RTE (required before other blocks)",
        "base_overrides": {"num_iters": 20, "population_size": 8},
        "variants": [
            {"sigma": s, "lr": a}
            for s, a in product(
                [3e-4, 1e-3, 3e-3, 1e-2],
                [1e-4, 3e-4, 1e-3, 3e-3],
            )
        ],
    },
    "one_vs_two": {
        "description": "One-sided vs two-sided (antithetic) ES",
        # One-sided uses half the per-iter fwd passes, so double iters to match budget.
        "base_overrides": {},
        "variants": [
            {"one_sided": False, "num_iters": BASE["num_iters"],      "label": "two_sided"},
            {"one_sided": True,  "num_iters": BASE["num_iters"] * 2,  "label": "one_sided"},
        ],
    },
    "noise_type": {
        "description": "Gaussian vs Rademacher perturbation noise",
        "base_overrides": {},
        "variants": [
            {"noise_type": "gaussian",    "label": "gaussian"},
            {"noise_type": "rademacher",  "label": "rademacher"},
        ],
    },
    "normalize": {
        "description": "Normalised vs unnormalised reward update",
        "base_overrides": {},
        "variants": [
            {"no_normalize": False, "label": "normalized"},
            {"no_normalize": True,  "label": "unnormalized"},
        ],
    },
    "top_k": {
        "description": "ES (all seeds) vs ARS-style top-k selection",
        "base_overrides": {"population_size": 16},  # need N>k; min top_k=2 for normalization
        "variants": [
            {"top_k": 0,  "label": "all_seeds"},
            {"top_k": 4,  "label": "top_k_4"},
            {"top_k": 8,  "label": "top_k_8"},
        ],
    },
    "pop_scaling": {
        "description": "Population size N ∈ {1,4,8,16,32} at fixed forward-pass budget",
        "base_overrides": {},
        "variants": [
            {"population_size": 1, "num_iters": _pop_iters(1), "no_normalize": True, "label": "N1"},
            *[{"population_size": n, "num_iters": _pop_iters(n), "label": f"N{n}"}
              for n in [4, 8, 16, 32]],
        ],
    },
    "mezo_pop_scaling": {
        "description": "MeZO: population size N ∈ {1,2,4,8} at fixed 20K-step budget",
        # Replicates MeZO paper (2305.17333) §3.2 + Appendix A on OPT-13B SST-2.
        # Paper hyperparameters: sigma=1e-3, lr=1e-6, train_size=1000, val_size=500,
        # 20K steps at n=1 (Algorithm 1). Appendix A ablations fix forward passes to
        # 10K; their Table 6 shows n=1/4/16 all within 1% at fixed forward-pass budget.
        #
        # Budget: 20K steps × 2 sides × batch=16 = 640K fwd passes for N=1.
        # Each variant is budget-matched so all consume 640K training forward passes.
        # val_every gives ~10 validation checkpoints per run.
        "base_overrides": {
            "task": "sst2",
            "model": "facebook/opt-13b",
            "train_size": 1000,
            "val_size": 500,
            "batch_size": 16,
            "sigma": 1e-3,
            "lr": 1e-7,
            "prompt_style": "mezo",
            "reward": "ce",
            "no_normalize": True,
            "no_save": True,
        },
        "variants": [
            {
                "population_size": n,
                # 640K fwd / (n * 2 sides * 16 batch) = iters to match 20K-step budget
                "num_iters": max(2, math.ceil(640_000 / (n * 2 * 16))),
                "val_every": max(1, math.ceil(max(2, math.ceil(640_000 / (n * 2 * 16))) / 10)),
                "label": f"N{n}",
            }
            for n in [1, 2, 4, 8]
        ],
    },
    "task_confirm": {
        "description": "Best calibrated config confirmed on BoolQ (set --best-sigma/--best-lr)",
        "base_overrides": {"task": "boolq", "val_size": 500},
        "variants": [
            # HPs are filled in at runtime from --best-sigma / --best-lr CLI args.
            # Defaults here match BASE and should be overridden after calibration.
            {"label": "boolq_best"},
        ],
    },
}


# ---------------------------------------------------------------------------
# CLI → train_es.py arg translation
# ---------------------------------------------------------------------------
_BOOL_FLAGS = {"one_sided": "--one-sided", "no_normalize": "--no-normalize", "no_save": "--no-save"}
_ARG_MAP = {
    "task":            "--task",
    "model":           "--model",
    "device":          "--device",
    "dtype":           "--dtype",
    "population_size": "--population-size",
    "num_iters":       "--num-iters",
    "batch_size":      "--batch-size",
    "val_every":       "--val-every",
    "sigma":           "--sigma",
    "lr":              "--lr",
    "train_size":      "--train-size",
    "val_size":        "--val-size",
    "early_stop_delta":"--early-stop-delta",
    "noise_type":      "--noise-type",
    "top_k":           "--top-k",
    "max_new_tokens":  "--max-new-tokens",
    "prompt_style":    "--prompt-style",
    "reward":          "--reward",
}


def config_to_cmd(cfg: dict, out_dir: Path, seed: int) -> list[str]:
    cmd = [sys.executable, "-m", "src.scripts.train_es", "--out-dir", str(out_dir), "--seed", str(seed)]
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
    parts = []
    if "label" in variant:
        return variant["label"]
    for k, v in variant.items():
        parts.append(f"{k[:4]}{v}")
    return "_".join(parts)


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
    base = {**BASE, **block["base_overrides"], **extra_base}
    if model:
        base["model"] = model

    variants = block["variants"]
    if extra_base.get("variant"):
        variants = [v for v in variants if variant_slug(v) == extra_base["variant"]]
        if not variants:
            raise SystemExit(f"[error] No variant matching '{extra_base['variant']}' in block '{block_name}'")
    print(f"\n{'='*60}")
    print(f"Block: {block_name} — {block['description']}")
    print(f"  {len(variants)} variant(s) × {n_seeds} seed(s) = {len(variants)*n_seeds} runs")
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
            run_seed = 42 + s
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
        print(f"  [{slug}] mean_best_val={mean_best} std_best_val={std_str}")

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
        print(f"  {rank}. {r['variant']:<20} mean={mv} std={sv}  seeds={seeds_str}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="ES variant experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--block",    default="calibration",
                   choices=list(BLOCKS.keys()) + ["all"],
                   help="Which experiment block to run")
    p.add_argument("--model",    default=None,
                   help="Override model for all runs (default: BASE['model'])")
    p.add_argument("--n-seeds",  type=int, default=3,
                   help="Number of independent seeds per variant")
    p.add_argument("--out-dir",  default=None,
                   help="Parent output dir (default: results/exp_<block>_<ts>)")
    p.add_argument("--device",   default=None,
                   help="Override device for all runs (e.g. cuda)")
    p.add_argument("--dtype",    default=None,
                   help="Override dtype for all runs")
    p.add_argument("--task",       default=None,
                   help="Override task for all runs")
    p.add_argument("--num-iters",      type=int, default=None,
                   help="Override num_iters for all runs")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override max_new_tokens for all runs")
    p.add_argument("--val-size",       type=int, default=None,
                   help="Override val_size for all runs")
    p.add_argument("--val-every",      type=int, default=None,
                   help="Override val_every for all runs")
    p.add_argument("--best-sigma", type=float, default=None,
                   help="Best sigma from calibration — applied to task_confirm block")
    p.add_argument("--best-lr",    type=float, default=None,
                   help="Best lr from calibration — applied to task_confirm block")
    p.add_argument("--variant",    default=None,
                   help="Run only the variant with this label (e.g. N32)")
    p.add_argument("--no-save",    action="store_true", default=False,
                   help="Disable checkpoint saving for all runs (saves disk space)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Block: {args.block}  |  n_seeds={args.n_seeds}\n"
        f"device={args.device}  dtype={args.dtype}  model={args.model}  task={args.task}\n"
        f"num_iters={args.num_iters}  val_every={args.val_every}  val_size={args.val_size}\n"
        f"best_sigma={args.best_sigma}  best_lr={args.best_lr}\n"
        f"variant={args.variant}  no_save={args.no_save}  out_dir={args.out_dir}"
    )

    extra_base = {}
    if args.device:
        extra_base["device"] = args.device
    if args.dtype:
        extra_base["dtype"] = args.dtype
    if args.task:
        extra_base["task"] = args.task
    if args.num_iters:
        extra_base["num_iters"] = args.num_iters
    if args.max_new_tokens:
        extra_base["max_new_tokens"] = args.max_new_tokens
    if args.val_size:
        extra_base["val_size"] = args.val_size
    if args.val_every:
        extra_base["val_every"] = args.val_every
    if args.no_save:
        extra_base["no_save"] = True
    if args.variant:
        extra_base["variant"] = args.variant

    # Apply calibrated HPs globally and patch task_confirm variants
    if args.best_sigma is not None:
        extra_base["sigma"] = args.best_sigma
    if args.best_lr is not None:
        extra_base["lr"] = args.best_lr
    if args.best_sigma is not None or args.best_lr is not None:
        patch: dict = {}
        if args.best_sigma is not None:
            patch["sigma"] = args.best_sigma
        if args.best_lr is not None:
            patch["lr"] = args.best_lr
        BLOCKS["task_confirm"]["variants"] = [{**BLOCKS["task_confirm"]["variants"][0], **patch}]
        print(f"[calibrated HPs applied globally] sigma={args.best_sigma}  lr={args.best_lr}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    blocks_to_run = list(BLOCKS.keys()) if args.block == "all" else [args.block]

    all_results = []
    for block_name in blocks_to_run:
        parent_dir = (
            Path(args.out_dir) / block_name
            if args.out_dir
            else Path("results") / f"exp_{block_name}_{ts}"
        )
        results = run_block(block_name, args.model, args.n_seeds, parent_dir, extra_base)
        print_block_summary(results)
        all_results.extend(results)

        parent_dir.mkdir(parents=True, exist_ok=True)
        summary_path = parent_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Summary saved → {summary_path}")

    if len(blocks_to_run) > 1:
        combined_path = Path(args.out_dir or f"results/exp_all_{ts}") / "combined_summary.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined summary → {combined_path}")


if __name__ == "__main__":
    main()
