"""Plot val_acc vs training forward passes for pop_scaling_1b, split into two panels.

Left panel:  small N (N=1, 2, 4, 8)
Right panel: large N (N=16, 32, 64, 128)

Usage:
  uv run python -m src.scripts.plot_pop_scaling_curves \\
      --primary   results_remote/exp_pop_scaling_1b_20260327_002340 \\
      --secondary results_remote/exp_pop_scaling_1b_large_n_20260327_031541 \\
      --out       results_remote/pop_scaling_1b_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SMALL_N = [1, 2, 4, 8]
LARGE_N = [16, 32, 64, 128]


def load_log(log_path: Path) -> list[dict]:
    entries = []
    for line in log_path.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def extract_curve(entries: list[dict]) -> tuple[list[int], list[float]]:
    fwds, accs = [], []
    for e in entries:
        if e.get("event") == "iter" and "val_acc" in e:
            x = e.get("train_fwd") or e.get("cumulative_fwd")
            if x is not None:
                fwds.append(x)
                accs.append(e["val_acc"])
    return fwds, accs


def collect_variant(variant_dir: Path) -> tuple[list, list, list]:
    """Return (fwds, mean_accs, std_accs) interpolated across seeds."""
    seed_curves = []
    for seed_dir in sorted(variant_dir.glob("seed*")):
        log = seed_dir / "log.jsonl"
        if not log.exists():
            continue
        entries = load_log(log)
        fwds, accs = extract_curve(entries)
        if fwds:
            seed_curves.append((np.array(fwds, dtype=float), np.array(accs)))

    if not seed_curves:
        return [], [], []
    if len(seed_curves) == 1:
        f, a = seed_curves[0]
        return f.tolist(), a.tolist(), [0.0] * len(a)

    grid_start = max(c[0][0]  for c in seed_curves)
    grid_end   = min(c[0][-1] for c in seed_curves)
    if grid_end <= grid_start:
        min_len = min(len(c[0]) for c in seed_curves)
        stacked = np.array([c[1][:min_len] for c in seed_curves])
        return seed_curves[0][0][:min_len].tolist(), stacked.mean(0).tolist(), stacked.std(0).tolist()

    n_pts = min(100, min(len(c[0]) for c in seed_curves))
    grid  = np.linspace(grid_start, grid_end, n_pts)
    interp = np.array([np.interp(grid, c[0], c[1]) for c in seed_curves])
    return grid.tolist(), interp.mean(0).tolist(), interp.std(0).tolist()


def discover_variants(exp_dirs: list[Path]) -> dict[str, Path]:
    """Collect variant_label -> variant_dir from multiple exp dirs, last one wins."""
    found = {}
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        for d in sorted(exp_dir.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "summary.json":
                found[d.name] = d
    return found


def parse_n(label: str) -> int:
    return int(label.lstrip("N"))


def plot_panel(ax, ns: list[int], variants: dict[str, Path], colors: list) -> None:
    for n, color in zip(ns, colors):
        label = f"N{n}"
        if label not in variants:
            continue
        fwds, means, stds = collect_variant(variants[label])
        if not fwds:
            continue
        fwds_k = [f / 1000 for f in fwds]
        means_a, stds_a = np.array(means), np.array(stds)
        line, = ax.plot(fwds_k, means_a, marker="o", markersize=3,
                        label=label, color=color, linewidth=1.8)
        ax.fill_between(fwds_k, means_a - stds_a, means_a + stds_a,
                        alpha=0.15, color=color)

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--primary",   required=True)
    p.add_argument("--secondary", default=None)
    p.add_argument("--out",       default="results_remote/pop_scaling_1b_curves.png")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dirs = [Path(args.primary)]
    if args.secondary:
        exp_dirs.append(Path(args.secondary))

    variants = discover_variants(exp_dirs)
    print(f"Found variants: {sorted(variants.keys(), key=parse_n)}")

    colors_small = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]
    colors_large = ["#984ea3", "#a65628", "#f781bf", "#999999"]

    fig, (ax_small, ax_large) = plt.subplots(1, 2, figsize=(13, 5))

    plot_panel(ax_small, SMALL_N, variants, colors_small)
    ax_small.set_title("Small N  (N = 1, 2, 4, 8)")

    plot_panel(ax_large, LARGE_N, variants, colors_large)
    ax_large.set_title("Large N  (N = 16, 32, 64, 128)")

    fig.suptitle("Population scaling — OPT-1.3B on BoolQ (fixed forward-pass budget)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
