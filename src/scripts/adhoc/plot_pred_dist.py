"""Plot stacked prediction distribution bars for base / latest / best checkpoints.

X-axis: base, latest, best
Within each: one grouped bar per gold label
Each bar is stacked by predicted class (colored segments).

Usage:
  uv run python -m src.scripts.plot_pred_dist \\
      --base    results/.../baseline_results.json \\
      --latest  results/.../baseline_results.json \\
      --best    results/.../baseline_results.json \\
      --task    sst2 \\
      [--out    pred_dist.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_result(path: str, task: str) -> dict:
    data = json.loads(Path(path).read_text())
    for entry in data:
        if entry.get("task") == task:
            return entry
    raise SystemExit(f"[error] Task '{task}' not found in {path}")


def pred_fractions(result: dict) -> dict[str, dict[str, float]]:
    pred_by_gold = result["pred_by_gold"]
    gold_dist = result["gold_dist"]
    return {
        gold: {pred: count / gold_dist[gold] for pred, count in preds.items()}
        for gold, preds in pred_by_gold.items()
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",   required=True)
    p.add_argument("--latest", required=True)
    p.add_argument("--best",   required=True)
    p.add_argument("--task",   required=True)
    p.add_argument("--out",    default=None)
    return p.parse_args()


def main():
    args = parse_args()

    model_names = ["base", "latest", "best"]
    paths = [args.base, args.latest, args.best]
    results = {n: load_result(p, args.task) for n, p in zip(model_names, paths)}
    fracs = {n: pred_fractions(r) for n, r in results.items()}

    gold_labels = sorted(set(g for f in fracs.values() for g in f))
    pred_labels = sorted(set(p for f in fracs.values() for gd in f.values() for p in gd))
    if "parse_fail" in pred_labels:
        pred_labels.remove("parse_fail")
        pred_labels.append("parse_fail")

    cmap = plt.get_cmap("tab10")
    pred_colors = {p: cmap(i) for i, p in enumerate(pred_labels)}

    n_models = len(model_names)
    n_gold = len(gold_labels)
    bar_width = 0.7 / n_gold
    group_spacing = 1.0
    group_centers = np.arange(n_models) * group_spacing

    # Offsets within each group for each gold label
    offsets = np.linspace(-(n_gold - 1) / 2, (n_gold - 1) / 2, n_gold) * bar_width

    fig, ax = plt.subplots(figsize=(3 * n_models + 2, 5))

    for gi, gold in enumerate(gold_labels):
        for mi, model in enumerate(model_names):
            x = group_centers[mi] + offsets[gi]
            frac = fracs[model].get(gold, {})
            bottom = 0.0
            for pred in pred_labels:
                h = frac.get(pred, 0.0)
                if h > 0:
                    ax.bar(x, h, width=bar_width, bottom=bottom,
                           color=pred_colors[pred], edgecolor="white", linewidth=0.5)
                    if h > 0.06:
                        ax.text(x, bottom + h / 2, f"{h:.0%}",
                                ha="center", va="center", fontsize=7, color="white", fontweight="bold")
                    bottom += h

    # X-axis labels with accuracy
    xtick_labels = [
        f"{n}\nacc={results[n]['accuracy']:.3f}"
        for n in model_names
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of gold examples")
    ax.set_title(f"Prediction distributions — task={args.task}")
    ax.grid(axis="y", alpha=0.3)

    # Legend: gold label hatches
    gold_legend = [
        mpatches.Patch(facecolor="white", edgecolor="gray",
                       label=f"bar position: gold={g}")
        for g in gold_labels
    ]
    pred_legend = [
        mpatches.Patch(facecolor=pred_colors[p], label=f"pred={p}")
        for p in pred_labels
    ]
    ax.legend(handles=gold_legend + pred_legend, loc="upper right",
              fontsize=8, framealpha=0.9)

    # Annotate gold label below each bar group
    for gi, gold in enumerate(gold_labels):
        for mi in range(n_models):
            x = group_centers[mi] + offsets[gi]
            ax.text(x, -0.05, gold, ha="center", va="top",
                    fontsize=7, color="dimgray", rotation=30)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
