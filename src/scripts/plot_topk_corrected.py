"""Single top-k plot with corrected top-k=2 (no normalization).

Sources:
  all_seeds, top_k_4, top_k_1 → exp_top_k_n8   (top_k=1 already uses no_normalize)
  top_k_2                     → exp_top_k_no_norm / top_k_2_nonorm  (replaces confounded run)

Usage:
  uv run python -m src.scripts.plot_topk_corrected \
      --n8-dir     results_remote/exp_top_k_n8_20260326_161554 \
      --nonorm-dir results_remote/exp_top_k_no_norm_20260326_175617 \
      --out        results_remote/top_k_corrected.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_curve(variant_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read all seed logs, interpolate to common grid, return (fwds_k, mean, std)."""
    seed_curves = []
    for seed_dir in sorted(variant_dir.glob("seed*")):
        log = seed_dir / "log.jsonl"
        if not log.exists():
            continue
        fwds, accs = [], []
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "iter" and "val_acc" in e:
                x = e.get("train_fwd") or e.get("cumulative_fwd")
                if x is not None:
                    fwds.append(x)
                    accs.append(e["val_acc"])
        if fwds:
            seed_curves.append((np.array(fwds, dtype=float), np.array(accs)))

    if not seed_curves:
        raise FileNotFoundError(f"No log data found in {variant_dir}")

    if len(seed_curves) == 1:
        f, a = seed_curves[0]
        return f / 1000, a, np.zeros_like(a)

    grid_start = max(c[0][0]  for c in seed_curves)
    grid_end   = min(c[0][-1] for c in seed_curves)
    n_pts = min(100, min(len(c[0]) for c in seed_curves))

    if grid_end <= grid_start:
        n_pts = min(len(c[0]) for c in seed_curves)
        stacked = np.array([c[1][:n_pts] for c in seed_curves])
        fwds_k  = seed_curves[0][0][:n_pts] / 1000
        return fwds_k, stacked.mean(0), stacked.std(0)

    grid = np.linspace(grid_start, grid_end, n_pts)
    interp = np.array([np.interp(grid, c[0], c[1]) for c in seed_curves])
    return grid / 1000, interp.mean(0), interp.std(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n8-dir",     required=True)
    p.add_argument("--nonorm-dir", required=True)
    p.add_argument("--out", default="results_remote/top_k_corrected.png")
    return p.parse_args()


def main():
    args  = parse_args()
    n8    = Path(args.n8_dir)
    nonorm = Path(args.nonorm_dir)

    # (display label, variant_dir, color)
    variants = [
        ("all seeds (k=8)",             n8    / "all_seeds",      "#377eb8"),
        ("top-k = 4",                   n8    / "top_k_4",        "#4daf4a"),
        ("top-k = 2  (no norm)",        nonorm / "top_k_2_nonorm", "#ff7f00"),
        ("top-k = 1  (no norm)",        n8    / "top_k_1",        "#e41a1c"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    baseline = 0.0939
    for label, vdir, color in variants:
        fwds_k, mean, std = load_curve(vdir)
        fwds_k = np.concatenate([[0.0], fwds_k])
        mean   = np.concatenate([[baseline], mean])
        std    = np.concatenate([[0.0],      std])
        ax.plot(fwds_k, mean, marker="o", markersize=3,
                label=label, color=color, linewidth=1.8)
        ax.fill_between(fwds_k, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Top-k ablation at N=8 — OPT-350M on BoolQ\n"
        "(top-k=2 uses corrected no-normalization run; fixed budget = 7,680 FPs)"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
