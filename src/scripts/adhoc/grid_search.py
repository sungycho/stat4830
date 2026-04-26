"""Hyperparameter grid/random search over the ES SST-2 loop.

Run with:
  uv run python -m src.scripts.grid_search                         # grid search, default config
  uv run python -m src.scripts.grid_search --mode random --n 6    # random search, 6 samples
  uv run python -m src.scripts.grid_search --config configs/my_grid.json

Each combination runs sanity_es_loop as a subprocess (fresh model weights per run).
All logs land in results/grid_<timestamp>/run_<i>/log.jsonl.
A summary table and summary.json are written at the end.

WARNING: runs are sequential on CPU. Keep the grid small or use --mode random.
Full default grid = 2×2×2×2 = 16 runs. With 4 iters each, expect several hours.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# Maps config-dict keys → CLI flags in sanity_es_loop.
# Add entries here when new CLI args are added to sanity_es_loop.
_ARG_MAP = {
    "task":            "--task",
    "population_size": "--population-size",
    "batch_size":      "--batch-size",
    "sigma":           "--sigma",
    "lr":              "--lr",
    "num_iters":       "--num-iters",
    "train_size":      "--train-size",
    "val_size":        "--val-size",
    "val_every":       "--val-every",
    "early_stop_delta":"--early-stop-delta",
    "max_new_tokens":  "--max-new-tokens",
}

DEFAULT_CONFIG = "configs/grid_search.json"


def parse_args():
    p = argparse.ArgumentParser(description="Grid/random search for ES on SST-2")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="JSON file defining the search space")
    p.add_argument("--mode", choices=["grid", "random"], default="grid")
    p.add_argument("--n", type=int, default=8,
                   help="Number of samples for --mode random")
    p.add_argument("--task", default=None,
                   help="Task for all runs (sst2/rte/boolq). Can also be swept via grid config.")
    p.add_argument("--model", default=None,
                   help="Model to use for all runs (overrides env MODEL_NAME)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed passed to every run")
    p.add_argument("--out-dir", default=None,
                   help="Parent output dir (default: results/grid_<timestamp>)")
    return p.parse_args()


def load_grid(config_path: str) -> dict:
    with open(config_path) as f:
        grid = json.load(f)
    # Strip comment keys
    return {k: v for k, v in grid.items() if not k.startswith("_")}


def generate_combinations(grid: dict, mode: str, n: int) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    if mode == "grid":
        return [dict(zip(keys, combo)) for combo in product(*values)]
    else:
        seen = set()
        combos = []
        attempts = 0
        while len(combos) < n and attempts < n * 10:
            attempts += 1
            combo = {k: random.choice(v) for k, v in grid.items()}
            key = json.dumps(combo, sort_keys=True)
            if key not in seen:
                seen.add(key)
                combos.append(combo)
        return combos


def combo_to_slug(combo: dict) -> str:
    """Short human-readable name for a hyperparameter combo."""
    parts = []
    for k, v in combo.items():
        short = k[:3]  # pop, bat, sig, lr_, num, tra
        parts.append(f"{short}{v}")
    return "_".join(parts)


def run_combination(combo: dict, run_dir: Path, model: str | None, task: str | None, seed: int) -> tuple[Path, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.scripts.train_es", "--out-dir", str(run_dir)]
    for key, val in combo.items():
        if key in _ARG_MAP:
            cmd.extend([_ARG_MAP[key], str(val)])
    if model:
        cmd.extend(["--model", model])
    if task:
        cmd.extend(["--task", task])
    cmd.extend(["--seed", str(seed)])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    return run_dir, result.returncode


def best_val_acc_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "iter" and "val_acc" in entry:
            if best is None or entry["val_acc"] > best:
                best = entry["val_acc"]
    return best


def print_summary(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda x: x["best_val_acc"] if x["best_val_acc"] is not None else -1,
        reverse=True,
    )
    col_w = 10
    header_combo = list(results[0]["combo"].keys()) if results else []

    header = "rank  best_val  status      " + "  ".join(f"{k:<{col_w}}" for k in header_combo) + "  run_dir"
    print("\n" + "=" * len(header))
    print("GRID SEARCH RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results, 1):
        val = f"{r['best_val_acc']:.3f}" if r["best_val_acc"] is not None else "N/A"
        status = "OK" if r["returncode"] == 0 else f"FAILED(rc={r['returncode']})"
        combo_str = "  ".join(f"{str(v):<{col_w}}" for v in r["combo"].values())
        print(f"{rank:<6}{val:<10}{status:<12}{combo_str}  {r['run_dir']}")
    print("=" * len(header))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    grid = load_grid(args.config)
    combos = generate_combinations(grid, args.mode, args.n)

    ts = time.strftime("%Y%m%d_%H%M%S")
    parent_dir = Path(args.out_dir) if args.out_dir else Path("results") / f"grid_{ts}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode} | {len(combos)} combinations | out: {parent_dir}")
    print(f"Search space: {grid}")

    t_start = time.perf_counter()
    results = []

    for i, combo in enumerate(combos):
        run_dir = parent_dir / f"run_{i:02d}_{combo_to_slug(combo)}"
        print(f"\n[{i+1}/{len(combos)}] {combo}")

        run_dir, returncode = run_combination(combo, run_dir, args.model, args.task, args.seed)

        log_path = run_dir / "log.jsonl"
        best = best_val_acc_from_log(log_path)
        if returncode != 0:
            print(f"  [!] Run failed with return code {returncode}")
        results.append({"combo": combo, "best_val_acc": best, "returncode": returncode, "run_dir": str(run_dir)})
        print(f"  → best_val_acc: {best}")

    # Save summary
    summary_path = parent_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"\nTotal wall-clock: {time.perf_counter() - t_start:.1f}s")
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
