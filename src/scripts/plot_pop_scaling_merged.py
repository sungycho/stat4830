"""Merge two pop_scaling summary.json files and plot combined results.

Usage:
  uv run python -m src.scripts.plot_pop_scaling_merged \\
      --primary   results_remote/exp_pop_scaling_1b_20260327_002340 \\
      --secondary results_remote/exp_pop_scaling_1b_large_n_20260327_031541 \\
      --out       results_remote/pop_scaling_1b_merged.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(exp_dir: Path) -> dict[str, dict]:
    """Return {variant_label: result_dict} from summary.json."""
    with open(exp_dir / "summary.json") as f:
        results = json.load(f)
    return {r["variant"]: r for r in results}


def merge(primary: dict, secondary: dict) -> dict:
    """Fill None entries in primary with values from secondary."""
    merged = dict(primary)
    for k, v in secondary.items():
        if k not in merged or merged[k]["mean_best_val"] is None:
            merged[k] = v
    return merged


def parse_n(label: str) -> int:
    return int(label.lstrip("N"))


def plot(merged: dict[str, dict], out_path: Path) -> None:
    # Sort by N
    items = sorted(merged.items(), key=lambda x: parse_n(x[0]))
    labels = [k for k, v in items if v["mean_best_val"] is not None]
    means  = [merged[k]["mean_best_val"] for k in labels]
    stds   = [merged[k]["std_best_val"] or 0.0 for k in labels]
    ns     = [parse_n(k) for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(ns, means, yerr=stds, fmt="o-", capsize=4,
                linewidth=2, markersize=6, color="steelblue", label="mean ± std")

    for n, m, s in zip(ns, means, stds):
        ax.annotate(f"{m:.3f}", (n, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Population size N  (log₂ scale)")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Population scaling — OPT-1.3B on BoolQ\n(fixed forward-pass budget)")
    ax.set_ylim(0, 1)
    ax.axhline(y=max(means), color="gray", linestyle="--", alpha=0.4, label=f"peak={max(means):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--primary",   required=True)
    p.add_argument("--secondary", required=True)
    p.add_argument("--out",       default="results_remote/pop_scaling_1b_merged.png")
    return p.parse_args()


def main():
    args = parse_args()
    primary   = load_summary(Path(args.primary))
    secondary = load_summary(Path(args.secondary))
    merged    = merge(primary, secondary)

    print("Merged results:")
    for k in sorted(merged, key=parse_n):
        v = merged[k]
        print(f"  {k}: mean={v['mean_best_val']}  std={v['std_best_val']}  seeds={v['best_per_seed']}")

    plot(merged, Path(args.out))


if __name__ == "__main__":
    main()
