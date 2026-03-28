"""Pop-scaling curves + RandOpt (Neural Thickets) reference lines.

Adds three horizontal dashed lines for RandOpt OPT-350M on BoolQ
(N=1000 population, varying K) and a base-model accuracy line to both
panels of the existing pop-scaling figure.

Usage:
  uv run python -m src.scripts.plot_pop_scaling_with_randopt \
      --primary   results_remote/exp_pop_scaling_1b_20260327_002340 \
      --secondary results_remote/exp_pop_scaling_1b_large_n_20260327_031541 \
      --out       results_remote/pop_scaling_1b_curves_with_randopt.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


SMALL_N = [1, 2, 4, 8]
LARGE_N = [16, 32, 64, 128]

# RandOpt (Neural Thickets) — OPT-350M on BoolQ, N=1000 population
RANDOPT = [
    {"k": 10,  "acc": 0.5498, "color": "#1b9e77", "ls": "--"},
    {"k": 50,  "acc": 0.4058, "color": "#d95f02", "ls": "--"},
    {"k": 100, "acc": 0.3893, "color": "#7570b3", "ls": "--"},
]
BASE_ACC = 0.1306   # OPT-350M BoolQ base model (no training)


# ---------------------------------------------------------------------------
# Data helpers (identical to plot_pop_scaling_curves.py)
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

    n_pts  = min(100, min(len(c[0]) for c in seed_curves))
    grid   = np.linspace(grid_start, grid_end, n_pts)
    interp = np.array([np.interp(grid, c[0], c[1]) for c in seed_curves])
    return grid.tolist(), interp.mean(0).tolist(), interp.std(0).tolist()


def discover_variants(exp_dirs: list[Path]) -> dict[str, Path]:
    found = {}
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        for d in sorted(exp_dir.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "summary.json":
                key = d.name.split("_")[0]   # "N2_nonorm" -> "N2"
                found[key] = d
    return found


def parse_n(label: str) -> int:
    return int(label.lstrip("N"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_randopt_lines(ax, x_max: float) -> None:
    """Draw RandOpt horizontal lines and base-model line across the panel."""
    ax.axhline(BASE_ACC, color="black", linestyle=":", linewidth=1.2, zorder=1)
    for r in RANDOPT:
        ax.axhline(r["acc"], color=r["color"], linestyle=r["ls"],
                   linewidth=1.3, zorder=1)


def plot_panel(ax, ns: list[int], variants: dict[str, Path],
               colors: list, x_max_ref: list) -> None:
    for n, color in zip(ns, colors):
        label = f"N{n}"
        if label not in variants:
            continue
        fwds, means, stds = collect_variant(variants[label])
        if not fwds:
            continue
        # prepend zero-shot baseline at x=0
        fwds_k  = [0.0] + [f / 1000 for f in fwds]
        means_a = np.array([BASE_ACC] + means)
        stds_a  = np.array([0.0]     + stds)
        line, = ax.plot(fwds_k, means_a, marker="o", markersize=3,
                        label=label, color=color, linewidth=1.8)
        ax.fill_between(fwds_k, means_a - stds_a, means_a + stds_a,
                        alpha=0.15, color=color)
        if fwds_k:
            x_max_ref[0] = max(x_max_ref[0], fwds_k[-1])

    add_randopt_lines(ax, x_max_ref[0])

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def build_legend_handles():
    handles = []
    handles.append(mlines.Line2D([], [], color="black", linestyle=":",
                                 linewidth=1.2, label=f"Base model ({BASE_ACC*100:.1f}%)"))
    for r in RANDOPT:
        handles.append(mlines.Line2D([], [], color=r["color"], linestyle=r["ls"],
                                     linewidth=1.3,
                                     label=f"RandOpt K={r['k']}, N=1000 ({r['acc']*100:.1f}%)"))
    return handles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--primary",   required=True)
    p.add_argument("--secondary", default=None)
    p.add_argument("--patch",     default=None, help="Extra dir applied last (its variants override earlier ones)")
    p.add_argument("--out", default="results_remote/pop_scaling_1b_curves_with_randopt.png")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dirs = [Path(args.primary)]
    if args.secondary:
        exp_dirs.append(Path(args.secondary))
    if args.patch:
        exp_dirs.append(Path(args.patch))

    variants = discover_variants(exp_dirs)
    print(f"Found variants: {sorted(variants.keys(), key=parse_n)}")

    colors_small = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]
    colors_large = ["#984ea3", "#a65628", "#f781bf", "#999999"]

    fig, (ax_small, ax_large) = plt.subplots(1, 2, figsize=(14, 5))

    x_max = [0.0]
    plot_panel(ax_small, SMALL_N, variants, colors_small, x_max)
    ax_small.set_title("Small N  (N = 1, 2, 4, 8)")

    x_max = [0.0]
    plot_panel(ax_large, LARGE_N, variants, colors_large, x_max)
    ax_large.set_title("Large N  (N = 16, 32, 64, 128)")

    # ES curve legend (auto) + RandOpt legend (manual) — merge into each panel
    randopt_handles = build_legend_handles()
    for ax in (ax_small, ax_large):
        es_handles, es_labels = ax.get_legend_handles_labels()
        ax.legend(handles=es_handles + randopt_handles,
                  labels=es_labels + [h.get_label() for h in randopt_handles],
                  loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Population scaling — OPT-1.3B on BoolQ (fixed forward-pass budget)\n"
        "with RandOpt (Neural Thickets) OPT-350M reference lines",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
