"""Plot decomposition categories over training (format/reasoning/regression thickets).

Reads decomp_log.jsonl produced by train_es.py --track-decomposition and plots
the four categories as stacked areas over training forward passes.

Usage:
  uv run python -m src.scripts.plot_decomp_tracking --run-dir results/failure_analysis/es_n8_tracked --out results/failure_analysis/decomp_tracking.png
  uv run python -m src.scripts.plot_decomp_tracking --run-dir results/failure_analysis/es_n8_tracked --run-dir2 results/failure_analysis/es_n64_tracked --labels "N=8" "N=64" --out results/failure_analysis/decomp_tracking.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


C_STRICT  = "#9E9E9E"
C_REASON  = "#64B5F6"
C_FORMAT  = "#CE93D8"
C_REGRESS = "#EF9A9A"


def load_decomp_log(run_dir: str) -> list[dict]:
    path = Path(run_dir) / "decomp_log.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"decomp_log.jsonl not found in {run_dir}. "
                                f"Did you run with --track-decomposition?")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot decomposition tracking from train_es.py --track-decomposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",  nargs="+", required=True,
                   help="One or more run directories containing decomp_log.jsonl")
    p.add_argument("--labels",   nargs="+", default=None,
                   help="Display labels for each run (same order as --run-dir)")
    p.add_argument("--out",      default="results/failure_analysis/decomp_tracking.png")
    p.add_argument("--dpi",      type=int, default=150)
    return p.parse_args()


def plot_single(ax, entries: list[dict], label: str, n: int):
    """Plot stacked area of 4 categories for a single run."""
    fwd     = [e["train_fwd"]        for e in entries]
    strict  = [100 * e["strictly_correct"]  / n for e in entries]
    reason  = [100 * e["reasoning_thicket"] / n for e in entries]
    fmt     = [100 * e["format_thicket"]    / n for e in entries]
    regress = [100 * e["regression"]        / n for e in entries]
    acc     = [100 * e["val_acc"]           for e in entries]

    # stacked area: strict | reasoning | format | regression (negative direction)
    ax.stackplot(fwd,
                 strict, reason, fmt,
                 labels=["Strictly Correct", "Reasoning Thicket", "Format Thicket"],
                 colors=[C_STRICT, C_REASON, C_FORMAT],
                 alpha=0.85)

    # regression as a separate line (it eats into the total)
    ax.plot(fwd, [-r for r in regress], color=C_REGRESS, linewidth=1.5,
            linestyle="--", label="Regression (−)")

    # net accuracy line
    ax.plot(fwd, acc, color="black", linewidth=1.8, label="Net val_acc", zorder=5)

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Training forward passes", fontsize=9)
    ax.set_ylabel("% of val set", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(-10, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_output_bias(ax, entries: list[dict], label: str, n: int):
    """Plot yes/no/format prediction distribution over training."""
    fwd        = [e["train_fwd"]             for e in entries]
    pred_yes   = [100 * e.get("pred_yes",    0) / n for e in entries]
    pred_no    = [100 * e.get("pred_no",     0) / n for e in entries]
    pred_fmt   = [100 * e.get("pred_format", 0) / n for e in entries]

    ax.stackplot(fwd, pred_yes, pred_no, pred_fmt,
                 labels=["Predicted Yes", "Predicted No", "Format Error"],
                 colors=["#81C784", "#E57373", "#BDBDBD"],
                 alpha=0.85)

    # majority-class line (true yes ratio)
    true_yes_pct = 100 * sum(1 for e in entries[:1]) / 1  # placeholder
    # draw 50% line for reference
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="50%")

    ax.set_title(f"{label} — output bias", fontsize=10)
    ax.set_xlabel("Training forward passes", fontsize=9)
    ax.set_ylabel("% of val set", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    args = parse_args()
    run_dirs = args.run_dir
    labels   = args.labels or [Path(d).name for d in run_dirs]
    assert len(labels) == len(run_dirs), "--labels must match number of --run-dir entries"

    all_entries = [load_decomp_log(d) for d in run_dirs]
    n = all_entries[0][0]["n"]

    ncols = len(run_dirs)
    # 2 rows: top = decomposition, bottom = output bias
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 8), squeeze=False)

    for col, (entries, label) in enumerate(zip(all_entries, labels)):
        plot_single(axes[0][col], entries, label, n)
        plot_output_bias(axes[1][col], entries, label, n)

    # legends
    handles0, lbls0 = axes[0][0].get_legend_handles_labels()
    fig.legend(handles0, lbls0, loc="lower center", ncol=len(handles0),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    handles1, lbls1 = axes[1][0].get_legend_handles_labels()
    fig.legend(handles1, lbls1, loc="lower center", ncol=len(handles1),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f"Decomposition tracking over training  (n={n})", fontsize=11)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
