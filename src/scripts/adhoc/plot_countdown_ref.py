"""Quick reference plot: Countdown ES curves for N=8 and N=64.

Usage:
  uv run python -m src.scripts.plot_countdown_ref \
      --n8-dir  results_remote/exp_countdown_N8_20260328_144152/N8 \
      --n64-dir results_remote/exp_countdown_N64_20260328_150329/N64 \
      --out     results_remote/countdown_N8_N64_ref.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_variant(variant_dir: Path):
    """Load all seed logs from a variant dir, return (fwds_k, mean, std)."""
    seed_dirs = sorted(variant_dir.glob("seed*"))
    all_curves: dict[float, list[float]] = {}
    baselines = []

    for sd in seed_dirs:
        log = sd / "log.jsonl"
        if not log.exists():
            continue
        baseline = None
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "baseline":
                baseline = e["val_acc"]
            if e.get("event") == "iter" and "val_acc" in e:
                x = e.get("train_fwd") or e.get("cumulative_fwd", 0)
                all_curves.setdefault(x, []).append(e["val_acc"])
        if baseline is not None:
            baselines.append(baseline)

    if not all_curves:
        return np.array([]), np.array([]), np.array([])

    baseline_mean = float(np.mean(baselines)) if baselines else 0.0
    xs = sorted(all_curves)
    means = [float(np.mean(all_curves[x])) for x in xs]
    stds  = [float(np.std(all_curves[x]))  for x in xs]

    fwds  = np.array([0.0] + [x / 1000 for x in xs])
    means = np.array([baseline_mean] + means)
    stds  = np.array([0.0] + stds)
    return fwds, means, stds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n8-dir",  required=True)
    p.add_argument("--n64-dir", required=True)
    p.add_argument("--out", default="results_remote/countdown_N8_N64_ref.png")
    return p.parse_args()


def main():
    args = parse_args()

    variants = [
        ("N=8",  Path(args.n8_dir),  "#377eb8"),
        ("N=64", Path(args.n64_dir), "#e41a1c"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, vdir, color in variants:
        fwds, means, stds = load_variant(vdir)
        if len(fwds) == 0:
            print(f"[warn] no data found in {vdir}")
            continue
        ax.plot(fwds, means, marker="o", markersize=3, linewidth=1.8,
                color=color, label=label)
        ax.fill_between(fwds, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 0.15)
    ax.set_title("Countdown — Qwen2.5-3B-Instruct ES (N=8 vs N=64)\n"
                 "Cold-start regime: model rarely produces valid expressions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
