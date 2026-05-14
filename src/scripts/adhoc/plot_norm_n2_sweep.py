"""Plot val_acc for a `run_norm_n2.py` sweep dir, averaged across seeds.

Auto-discovers `norm_on_s*/log.jsonl` and `norm_off_s*/log.jsonl` under the
sweep dir, interpolates each seed onto a common forward-pass grid, and
overlays the two conditions with mean ± std bands.

Usage:
  uv run python -m src.scripts.adhoc.plot_norm_n2_sweep \
      --exp-dir results/norm_n2_opt-1.3b_cb_20260508_024622
  uv run python -m src.scripts.adhoc.plot_norm_n2_sweep \
      --exp-dir results/norm_n2_opt-1.3b_cb_20260508_024622 \
      --out results/norm_n2_opt-1.3b_cb_20260508_024622/norm_comparison.png \
      --title "OPT-1.3B — CB N=2: normalize on vs off"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path) -> tuple[list[int], list[float], float | None]:
    fwds, accs = [], []
    baseline = None
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "baseline" and baseline is None:
            baseline = d.get("val_acc")
        if d.get("event") == "iter" and "val_acc" in d:
            x = d.get("train_fwd") or d.get("cumulative_fwd")
            if x is not None:
                fwds.append(x)
                accs.append(d["val_acc"])
    return fwds, accs, baseline


def find_seed_logs(exp_dir: Path, prefix: str) -> list[Path]:
    return sorted((exp_dir / d.name) / "log.jsonl"
                  for d in exp_dir.iterdir()
                  if d.is_dir() and d.name.startswith(prefix) and (d / "log.jsonl").exists())


def average_seeds(log_paths: list[Path], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    curves = []
    for path in log_paths:
        fwds, accs, baseline = load_log(path)
        if not fwds:
            continue
        if baseline is not None:
            fwds = [0] + fwds
            accs = [baseline] + accs
        fwds_a = np.array(fwds, dtype=float)
        accs_a = np.array(accs, dtype=float)
        interp = np.interp(grid, fwds_a, accs_a, right=np.nan)
        interp[grid > fwds_a[-1]] = np.nan
        curves.append(interp)

    if not curves:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan), 0

    stacked = np.array(curves)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), len(curves)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True, type=Path,
                   help="Sweep parent dir from run_norm_n2.py")
    p.add_argument("--out", default=None,
                   help="Output PNG (default: <exp-dir>/norm_comparison.png)")
    p.add_argument("--title", default=None, help="Plot title override")
    p.add_argument("--n-pts", type=int, default=100)
    args = p.parse_args()

    on_logs  = find_seed_logs(args.exp_dir, "norm_on_s")
    off_logs = find_seed_logs(args.exp_dir, "norm_off_s")
    if not on_logs and not off_logs:
        raise SystemExit(f"No norm_on_s*/log.jsonl or norm_off_s*/log.jsonl under {args.exp_dir}")

    # Common grid: 0 → min(last fwd) across all available seeds
    last_fwds = []
    for path in on_logs + off_logs:
        fwds, _, _ = load_log(path)
        if fwds:
            last_fwds.append(fwds[-1])
    grid_max = min(last_fwds) if last_fwds else 1
    grid = np.linspace(0, grid_max, args.n_pts)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label_prefix, logs, color in [
        ("normalize=off", off_logs, "tomato"),
        ("normalize=on",  on_logs,  "steelblue"),
    ]:
        if not logs:
            continue
        mean, std, n = average_seeds(logs, grid)
        ax.plot(grid, mean, color=color, linewidth=2,
                label=f"{label_prefix} ({n} seed{'s' if n != 1 else ''})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(args.title or f"{args.exp_dir.name}: normalize on vs off (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out) if args.out else args.exp_dir / "norm_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
