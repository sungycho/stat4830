"""Accuracy decomposition plot (Neural Thickets style).

Compares a base model against one or more methods and decomposes accuracy into:
  - Strictly Correct : base correct → method correct
  - Reasoning Thicket: base wrong answer → method correct
  - Format Thicket   : base format error → method correct
  - Regression       : base correct → method wrong/format_error

Requires per-example JSON files produced by analyze_failures.py --out.

Usage:
  # 1. First produce base model JSON:
  uv run python -m src.scripts.analyze_failures \
      --model facebook/opt-350m --val-size 500 \
      --out results/failure_analysis/base.json

  # 2. Produce method JSON(s) via checkpoint:
  uv run python -m src.scripts.analyze_failures \
      --model facebook/opt-350m --val-size 500 \
      --checkpoint results/exp_week9/task_confirm/boolq_best/seed0/best.pt \
      --out results/failure_analysis/es_seed0.json

  # 3. Plot:
  uv run python -m src.scripts.plot_failure_decomposition \
      --base results/failure_analysis/base.json \
      --methods results/failure_analysis/es_seed0.json \
      --labels "ES (seed 0)" \
      --out results/failure_analysis/decomposition.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── colours matching the reference figure ───────────────────────────────────
C_STRICT   = "#9E9E9E"   # grey   — strictly correct
C_REASON   = "#64B5F6"   # blue   — reasoning thicket
C_FORMAT   = "#CE93D8"   # purple — format thicket
C_REGRESS  = "#EF9A9A"   # red    — regression (shown as negative / hatched)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def decompose(base_examples: list[dict], method_examples: list[dict]) -> dict:
    """Compute per-example transition counts."""
    assert len(base_examples) == len(method_examples), \
        "Base and method must have the same number of examples (same val seed)."

    strictly_correct = 0
    reasoning_thicket = 0
    format_thicket = 0
    regression = 0
    # examples that were wrong in both — not plotted but tracked
    still_wrong = 0

    for b, m in zip(base_examples, method_examples):
        base_cat   = b["category"]
        method_cat = m["category"]

        base_ok   = base_cat   == "correct"
        method_ok = method_cat == "correct"

        if base_ok and method_ok:
            strictly_correct += 1
        elif base_ok and not method_ok:
            regression += 1
        elif not base_ok and method_ok:
            if base_cat == "format_error":
                format_thicket += 1
            else:  # wrong_answer
                reasoning_thicket += 1
        else:
            still_wrong += 1

    total = len(base_examples)
    net_correct = strictly_correct + reasoning_thicket + format_thicket - regression

    return {
        "strictly_correct":  strictly_correct,
        "reasoning_thicket": reasoning_thicket,
        "format_thicket":    format_thicket,
        "regression":        regression,
        "still_wrong":       still_wrong,
        "total":             total,
        "net_correct":       net_correct,
        "net_acc":           net_correct / total,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Neural-Thickets-style accuracy decomposition plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base",    required=True,
                   help="JSON from analyze_failures.py for the base model")
    p.add_argument("--methods", nargs="+", required=True,
                   help="JSON file(s) for each method to compare")
    p.add_argument("--labels",  nargs="+", default=None,
                   help="Display names for each method (same order as --methods)")
    p.add_argument("--out",     default="results/failure_analysis/decomposition.png")
    p.add_argument("--dpi",     type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    base_data = load_json(args.base)
    base_examples = base_data["examples"]
    n = len(base_examples)

    # Base model row: everything is strictly correct or not, no transitions
    base_correct = sum(1 for e in base_examples if e["category"] == "correct")
    base_format  = sum(1 for e in base_examples if e["category"] == "format_error")
    base_wrong   = sum(1 for e in base_examples if e["category"] == "wrong_answer")
    base_acc     = base_correct / n

    labels = args.labels or [Path(m).stem for m in args.methods]
    assert len(labels) == len(args.methods), "--labels must match number of --methods"

    # ── compute decompositions ───────────────────────────────────────────────
    rows = []  # one per method
    for method_path, label in zip(args.methods, labels):
        method_data = load_json(method_path)
        d = decompose(base_examples, method_data["examples"])
        d["label"] = label
        d["model"] = method_data.get("model", "")
        d["checkpoint"] = method_data.get("checkpoint", "")
        rows.append(d)

    # ── print table ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Method':<20} {'Strict':>7} {'Reason':>7} {'Format':>7} {'Regress':>8} {'Net acc':>8}")
    print(f"{'Base model':<20} {base_correct:>7} {'0':>7} {'0':>7} {'0':>8} {base_acc:>8.3f}")
    for r in rows:
        print(f"{r['label']:<20} {r['strictly_correct']:>7} {r['reasoning_thicket']:>7} "
              f"{r['format_thicket']:>7} {r['regression']:>8} {r['net_acc']:>8.3f}")
    print(f"{'='*72}")
    print(f"n={n}  |  base: {base_correct} correct / {base_format} format_err / {base_wrong} wrong")

    # ── plot ─────────────────────────────────────────────────────────────────
    row_labels = ["Base\nModel"] + [r["label"] for r in rows]
    n_rows = len(row_labels)

    fig, ax = plt.subplots(figsize=(10, 1.2 * n_rows + 1.5))

    bar_h = 0.55
    y_positions = np.arange(n_rows)[::-1]  # top to bottom

    def pct(count): return 100 * count / n

    for i, (y, label) in enumerate(zip(y_positions, row_labels)):
        if i == 0:
            # Base model row
            sc = pct(base_correct)
            ax.barh(y, sc, height=bar_h, color=C_STRICT, left=0)
            count_str = f"{base_correct} + 0 + 0 = {base_correct}"
            ax.text(sc + 0.5, y, f"{base_acc*100:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold")
            ax.text(0.5, y - bar_h * 0.55, count_str,
                    va="top", ha="left", fontsize=7, color="#555555")
        else:
            r = rows[i - 1]
            sc = pct(r["strictly_correct"])
            rt = pct(r["reasoning_thicket"])
            ft = pct(r["format_thicket"])
            rg = pct(r["regression"])

            # stacked: strictly_correct | reasoning_thicket | format_thicket
            ax.barh(y, sc, height=bar_h, color=C_STRICT,  left=0)
            ax.barh(y, rt, height=bar_h, color=C_REASON,  left=sc)
            ax.barh(y, ft, height=bar_h, color=C_FORMAT,  left=sc + rt)

            # regression shown as a hatched negative-looking bar after the stack
            if rg > 0:
                ax.barh(y, rg, height=bar_h, color=C_REGRESS,
                        left=sc + rt + ft, hatch="//", edgecolor="white", linewidth=0.5)

            count_str = (f"{r['strictly_correct']} + {r['reasoning_thicket']} + "
                         f"{r['format_thicket']} \u2212 {r['regression']} = {r['net_correct']}")
            ax.text(sc + rt + ft + rg + 0.5, y, f"{r['net_acc']*100:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold")
            ax.text(0.5, y - bar_h * 0.55, count_str,
                    va="top", ha="left", fontsize=7, color="#555555")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.8, n_rows - 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # vertical dashed line at base model accuracy
    ax.axvline(base_acc * 100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # legend
    legend_handles = [
        mpatches.Patch(color=C_STRICT,  label="Strictly Correct (correct answer & format)"),
        mpatches.Patch(color=C_FORMAT,  label="Format Thicket (format fixed, then correct)"),
        mpatches.Patch(color=C_REASON,  label="Reasoning Thicket (base wrong, method corrects)"),
        mpatches.Patch(color=C_REGRESS, label="Regression (base correct, method changes to wrong)",
                       hatch="//", edgecolor="white"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
              framealpha=0.9, ncol=2)

    task_info = base_data.get("model", "")
    ax.set_title(f"Accuracy Decomposition — BoolQ  ({task_info})", fontsize=11, pad=10)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\n[plot] saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
