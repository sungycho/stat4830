"""Plot norm-on vs norm-off N=2 val_acc for any sweep dir with the layout:
    <sweep>/norm_{on,off}_s{42,43,44}/log.jsonl

Usage:
  uv run python -m src.scripts.adhoc.plot_norm_sweep <sweep_dir> [--out out.png]
  uv run python -m src.scripts.adhoc.plot_norm_sweep --all          # plot every results/norm_n2_*
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED_SUFFIXES = ["s42", "s43", "s44"]
N_PTS = 100


def load_log(path: Path) -> tuple[list[int], list[float], float | None]:
    fwds, accs, baseline = [], [], None
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


def average_seeds(log_paths: list[Path], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for p in log_paths:
        fwds, accs, baseline = load_log(p)
        if not fwds:
            continue
        if baseline is not None:
            fwds = [0] + fwds
            accs = [baseline] + accs
        f = np.array(fwds, dtype=float)
        a = np.array(accs, dtype=float)
        interp = np.interp(grid, f, a, right=np.nan)
        interp[grid > f[-1]] = np.nan
        curves.append(interp)
    stacked = np.array(curves)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def read_config(path: Path) -> dict:
    with path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "config":
                return d
    return {}


def plot_sweep(sweep: Path, out: Path) -> None:
    on_logs = [sweep / f"norm_on_{s}" / "log.jsonl" for s in SEED_SUFFIXES]
    off_logs = [sweep / f"norm_off_{s}" / "log.jsonl" for s in SEED_SUFFIXES]
    on_logs = [p for p in on_logs if p.exists()]
    off_logs = [p for p in off_logs if p.exists()]
    if not on_logs or not off_logs:
        print(f"skip {sweep.name}: missing logs (on={len(on_logs)}, off={len(off_logs)})")
        return

    cfg = read_config(on_logs[0])
    model = cfg.get("model", "?")
    task = cfg.get("task", "?")
    N = cfg.get("population_size", "?")
    sigma = cfg.get("sigma", "?")
    lr = cfg.get("lr", "?")
    B = cfg.get("batch_size", "?")

    grid_max = 0
    for p in on_logs + off_logs:
        f, _, _ = load_log(p)
        if f:
            grid_max = max(grid_max, f[-1])
    if grid_max == 0:
        print(f"skip {sweep.name}: no train_fwd data")
        return
    grid = np.linspace(0, grid_max, N_PTS)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, logs, color in [
        (f"normalize=off ({len(off_logs)} seeds)", off_logs, "tomato"),
        (f"normalize=on ({len(on_logs)} seeds)", on_logs, "steelblue"),
    ]:
        mean, std = average_seeds(logs, grid)
        ax.plot(grid, mean, color=color, linewidth=2, label=label)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color)

    short_model = model.split("/")[-1]
    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"{short_model} — {task} N={N}: normalize on vs off "
                 f"(σ={sigma}, lr={lr}, B={B})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep", nargs="?", help="path to a single norm_n2_* dir")
    p.add_argument("--out", default=None)
    p.add_argument("--all", action="store_true", help="plot every results/norm_n2_*")
    args = p.parse_args()

    if args.all:
        sweeps = sorted(Path(s) for s in glob.glob("results/norm_n2_*") if Path(s).is_dir())
        for sd in sweeps:
            print(f"== {sd.name}")
            plot_sweep(sd, sd / "norm_comparison.png")
        return

    if not args.sweep:
        print("usage: plot_norm_sweep <sweep_dir> | --all", file=sys.stderr)
        sys.exit(2)
    sd = Path(args.sweep)
    out = Path(args.out) if args.out else sd / "norm_comparison.png"
    plot_sweep(sd, out)


if __name__ == "__main__":
    main()
