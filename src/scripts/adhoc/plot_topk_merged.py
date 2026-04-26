"""Create merged two-panel top-k figure for Week 10 report.

Left panel:  N=8 top-k ablation (all_seeds / top_k_4 / top_k_1 — top_k_2 with
             normalization ON is excluded because z-score over 2 samples is always ±1,
             producing a spurious advantage signal).
Right panel: Normalization confound control — top_k_2 and top_k_1 *both* with
             normalization OFF, showing they perform identically once the confound
             is removed.

Usage:
  uv run python -m src.scripts.plot_topk_merged \
      --n8-dir    results_remote/exp_top_k_n8_20260326_161554 \
      --nonorm-dir results_remote/exp_top_k_no_norm_20260326_175617 \
      --out        results_remote/top_k_merged.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data helpers (same pattern as plot_pop_scaling_curves.py)
# ---------------------------------------------------------------------------

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


def plot_panel(ax, variants: list[tuple[str, Path, str]], title: str) -> None:
    """variants = list of (display_label, variant_dir, color)."""
    for label, vdir, color in variants:
        if not vdir.exists():
            print(f"  WARNING: {vdir} not found, skipping {label}")
            continue
        fwds, means, stds = collect_variant(vdir)
        if not fwds:
            print(f"  WARNING: no data for {label}")
            continue
        fwds_k = [f / 1000 for f in fwds]
        means_a, stds_a = np.array(means), np.array(stds)
        ax.plot(fwds_k, means_a, marker="o", markersize=3,
                label=label, color=color, linewidth=1.8)
        ax.fill_between(fwds_k, means_a - stds_a, means_a + stds_a,
                        alpha=0.15, color=color)

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.4, 0.75)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n8-dir",    required=True, help="exp_top_k_n8_* directory")
    p.add_argument("--nonorm-dir",required=True, help="exp_top_k_no_norm_* directory")
    p.add_argument("--out", default="results_remote/top_k_merged.png")
    return p.parse_args()


def main():
    args = parse_args()
    n8   = Path(args.n8_dir)
    nonorm = Path(args.nonorm_dir)

    # Left panel: N=8 ablation — drop top_k_2 (normalization confound)
    n8_variants = [
        ("all seeds (k=N=8)", n8 / "all_seeds",  "#377eb8"),
        ("top-k=4",           n8 / "top_k_4",    "#4daf4a"),
        ("top-k=1 (no norm)", n8 / "top_k_1",    "#e41a1c"),
    ]

    # Right panel: normalization confound control (both without norm)
    nonorm_variants = [
        ("top-k=2, no norm", nonorm / "top_k_2_nonorm", "#ff7f00"),
        ("top-k=1, no norm", nonorm / "top_k_1_nonorm", "#e41a1c"),
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    plot_panel(ax_left,  n8_variants,
               "Top-k ablation at N=8\n(top-k=2 with normalization excluded — see right panel)")
    plot_panel(ax_right, nonorm_variants,
               "Normalization confound control (N=8)\ntop-k=1 vs top-k=2, both without z-score normalization")

    fig.suptitle(
        "Top-k selection ablation — OPT-350M on BoolQ  (fixed forward-pass budget = 7,680)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
