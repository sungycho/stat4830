"""Plot prediction distribution evolution over training checkpoints.

One subplot per gold label. Within each subplot, x-axis = checkpoints
(baseline + val_every steps), bars stacked by predicted label.

Usage:
  uv run python -m src.scripts.plot_pred_dist_evolution \\
      --log results/.../log.jsonl \\
      [--out evolution.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_checkpoints(log_path: str | Path) -> list[dict]:
    """Read log.jsonl and return all entries with pred_by_gold, sorted by train_fwd."""
    checkpoints = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "pred_by_gold" in entry:
                checkpoints.append(entry)
    checkpoints.sort(key=lambda e: e.get("train_fwd", -1))
    return checkpoints


def plot_evolution(log_path: str | Path, out_path: str | Path | None = None) -> None:
    checkpoints = load_checkpoints(log_path)
    if not checkpoints:
        print("[warn] No pred_by_gold entries found in log. Skipping evolution plot.")
        return

    gold_labels = sorted(set(g for c in checkpoints for g in c["pred_by_gold"]))
    pred_labels = sorted(set(p for c in checkpoints for gd in c["pred_by_gold"].values() for p in gd))
    if "parse_fail" in pred_labels:
        pred_labels.remove("parse_fail")
        pred_labels.append("parse_fail")

    cmap = plt.get_cmap("tab10")
    pred_colors: dict = {p: cmap(i) for i, p in enumerate(pred_labels)}
    if "parse_fail" in pred_colors:
        pred_colors["parse_fail"] = (0.65, 0.65, 0.65, 1.0)

    n_ckpts = len(checkpoints)
    n_gold = len(gold_labels)
    bar_width = 0.7
    xs = np.arange(n_ckpts)

    # One subplot per gold label, side by side
    panel_w = max(6, 0.45 * n_ckpts)
    fig, axes = plt.subplots(1, n_gold, figsize=(panel_w * n_gold, 5), sharey=True)
    if n_gold == 1:
        axes = [axes]

    run_name = Path(log_path).parent.name

    for gi, (gold, ax) in enumerate(zip(gold_labels, axes)):
        for ci, ckpt in enumerate(checkpoints):
            preds = ckpt["pred_by_gold"].get(gold, {})
            total = ckpt["gold_dist"].get(gold, 1)
            bottom = 0.0
            for pred in pred_labels:
                h = preds.get(pred, 0) / total
                if h <= 0:
                    continue
                ax.bar(xs[ci], h, width=bar_width, bottom=bottom,
                       color=pred_colors[pred], edgecolor="white", linewidth=0.3)
                if h > 0.07:
                    ax.text(xs[ci], bottom + h / 2, f"{h:.0%}",
                            ha="center", va="center", fontsize=6, color="white", fontweight="bold")
                bottom += h

        # X-axis: checkpoint labels (iteration + val_acc)
        xtick_labels = []
        for ckpt in checkpoints:
            event = ckpt.get("event", "")
            label = "base" if event == "baseline" else f"i{ckpt.get('iteration', '')}"
            acc = ckpt.get("val_acc", float("nan"))
            xtick_labels.append(f"{label}\n{acc:.2f}")

        ax.set_xticks(xs)
        ax.set_xticklabels(xtick_labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(f"gold = {gold}", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if gi == 0:
            ax.set_ylabel("Fraction of examples")

    pred_legend = [mpatches.Patch(facecolor=pred_colors[p], label=f"pred={p}") for p in pred_labels]
    fig.legend(handles=pred_legend, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Prediction distribution evolution — {run_name}", fontsize=12, y=1.01)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot prediction distribution evolution from a train_es.py log.jsonl"
    )
    p.add_argument("--log", required=True, help="Path to log.jsonl")
    p.add_argument("--out", default=None, help="Output image path (default: show interactively)")
    return p.parse_args()


def main():
    args = parse_args()
    plot_evolution(args.log, args.out)


if __name__ == "__main__":
    main()
