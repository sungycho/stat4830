"""Plot individual seed curves for N=1 on OPT-1.3B to show variance source.

Usage:
  uv run python -m src.scripts.plot_n1_seeds \
      --variant-dir results_remote/exp_pop_scaling_1b_20260327_002340/N1 \
      --out         results_remote/n1_seeds.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_ACC = 0.0939
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]


def load_curve(log_path: Path):
    fwds, accs = [], []
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "iter" and "val_acc" in e:
            x = e.get("train_fwd") or e.get("cumulative_fwd")
            if x is not None:
                fwds.append(x / 1000)
                accs.append(e["val_acc"])
    return fwds, accs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--out", default="results_remote/n1_seeds.png")
    return p.parse_args()


def main():
    args = parse_args()
    variant_dir = Path(args.variant_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    seed_dirs = sorted(variant_dir.glob("seed*"))
    for i, seed_dir in enumerate(seed_dirs):
        log = seed_dir / "log.jsonl"
        if not log.exists():
            continue
        fwds, accs = load_curve(log)
        seed_num = 42 + i
        color = COLORS[i % len(COLORS)]

        # prepend baseline
        fwds = [0.0] + fwds
        accs = [BASE_ACC] + accs

        peak = max(accs)
        ax.plot(fwds, accs, marker="o", markersize=3, linewidth=1.8,
                color=color, label=f"seed {seed_num}  (peak={peak:.3f})")

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    variant_name = variant_dir.name
    ax.set_title(f"{variant_name} individual seeds — OPT-1.3B on BoolQ\n(variance across random initializations)")
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
