"""Visualize the distribution of ES advantages per top-k variant.

Produces two figures:
  1. advantage_dist.png  -- violin/box plots of |advantage| per variant,
                            aggregated across all seeds and iterations.
                            Shows how spread-out the reward signal is in each setting.
  2. advantage_cv.png    -- coefficient of variation (std/mean of |A_i|) per
                            iteration over training, mean ± std across seeds.
                            Shows how the landscape "informativeness" evolves.

Usage:
  uv run python -m src.scripts.plot_advantage_dist \\
      --exp-dir results_remote/exp_top_k_n8_20260326_161554
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
# Data loading
# ---------------------------------------------------------------------------

def load_advantages(log_path: Path) -> tuple[list[list[float]], list[int]]:
    """Return (advantages_per_iter, train_fwd_per_iter) from a log.jsonl."""
    adv_per_iter, fwd_per_iter = [], []
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "iter" and "advantages" in e:
            adv_per_iter.append(e["advantages"])
            fwd_per_iter.append(e.get("train_fwd", len(adv_per_iter) * 256))
    return adv_per_iter, fwd_per_iter


def collect_variant(variant_dir: Path) -> dict:
    """Collect advantages across all seeds for one variant."""
    all_abs_flat = []       # all |A_i| values, flattened across seeds+iters
    cv_per_seed = []        # CV curve per seed: list of per-iter CV values
    fwd_per_seed = []       # x-axis (train_fwd) per seed

    for seed_dir in sorted(variant_dir.glob("seed*")):
        log = seed_dir / "log.jsonl"
        if not log.exists():
            continue
        adv_iters, fwds = load_advantages(log)
        if not adv_iters:
            continue

        abs_flat = [abs(a) for itr in adv_iters for a in itr]
        all_abs_flat.extend(abs_flat)

        cvs = []
        for itr in adv_iters:
            abs_itr = [abs(a) for a in itr]
            mean_ = np.mean(abs_itr)
            std_  = np.std(abs_itr)
            cv = std_ / mean_ if mean_ > 1e-9 else 0.0
            cvs.append(cv)
        cv_per_seed.append(cvs)
        fwd_per_seed.append(fwds)

    return {
        "all_abs_flat": all_abs_flat,
        "cv_per_seed":  cv_per_seed,
        "fwd_per_seed": fwd_per_seed,
    }


# ---------------------------------------------------------------------------
# Plot 1: violin + box of |advantage| per variant
# ---------------------------------------------------------------------------

def plot_dist(data: dict[str, dict], out_path: Path) -> None:
    """Violin plot of pooled |advantage| across all seeds and iterations."""
    labels = list(data.keys())
    values = [data[k]["all_abs_flat"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(values, positions=range(len(labels)),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)

    # Overlay box
    ax.boxplot(values, positions=range(len(labels)),
               widths=0.15, patch_artist=False,
               medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(linewidth=1),
               capprops=dict(linewidth=1),
               flierprops=dict(marker=".", markersize=2, alpha=0.3))

    # Annotate median + CV
    for i, v in enumerate(values):
        med = np.median(v)
        cv  = np.std(v) / np.mean(v) if np.mean(v) > 1e-9 else 0
        ax.text(i, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.3,
                f"med={med:.3f}\nCV={cv:.2f}",
                ha="center", va="top", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Variant (top-k setting)")
    ax.set_ylabel("|Advantage|")
    ax.set_title("Distribution of |advantage| per top-k variant\n(pooled across all seeds × iterations)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: CV over training per variant
# ---------------------------------------------------------------------------

def _interpolate_to_grid(cv_per_seed, fwd_per_seed, n_points=50):
    """Interpolate per-seed CV curves onto a common forward-pass grid."""
    if not cv_per_seed:
        return [], [], []
    grid_start = max(s[0]  for s in fwd_per_seed)
    grid_end   = min(s[-1] for s in fwd_per_seed)
    if grid_end <= grid_start:
        min_len = min(len(s) for s in cv_per_seed)
        stacked = np.array([s[:min_len] for s in cv_per_seed])
        return fwd_per_seed[0][:min_len], stacked.mean(0).tolist(), stacked.std(0).tolist()
    grid = np.linspace(grid_start, grid_end, n_points)
    interpolated = np.array([
        np.interp(grid, fwds, cvs)
        for cvs, fwds in zip(cv_per_seed, fwd_per_seed)
    ])
    return grid.tolist(), interpolated.mean(0).tolist(), interpolated.std(0).tolist()


def plot_cv(data: dict[str, dict], out_path: Path) -> None:
    """CV of |advantage| over training — mean ± std across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, d in data.items():
        fwds, means, stds = _interpolate_to_grid(d["cv_per_seed"], d["fwd_per_seed"])
        if not fwds:
            continue
        fwds_k = [f / 1000 for f in fwds]
        means_a, stds_a = np.array(means), np.array(stds)
        line, = ax.plot(fwds_k, means_a, marker="o", markersize=3, label=label)
        ax.fill_between(fwds_k, means_a - stds_a, means_a + stds_a,
                        alpha=0.2, color=line.get_color())

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("CV of |advantage|  (std / mean)")
    ax.set_title(
        "Landscape informativeness over training\n"
        "High CV → reward signal is directional  |  Low CV → landscape is flat/isotropic"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Raw advantage distributions early vs late training (per variant)
# ---------------------------------------------------------------------------

def plot_early_vs_late(data: dict[str, dict], out_path: Path) -> None:
    """Histogram of |advantage| in first third vs last third of training."""
    variants = list(data.keys())
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, variants):
        d = data[label]
        early, late = [], []
        for cv_list, fwd_list, seed_dir in zip(
            d["cv_per_seed"], d["fwd_per_seed"],
            [""] * len(d["cv_per_seed"])   # placeholder
        ):
            pass  # we need raw per-iter advantages, stored separately below

        # Re-load raw advantages split by early/late
        variant_dir = data[label]["_dir"]
        all_early, all_late = [], []
        for seed_subdir in sorted(variant_dir.glob("seed*")):
            log = seed_subdir / "log.jsonl"
            if not log.exists():
                continue
            adv_iters, _ = load_advantages(log)
            if not adv_iters:
                continue
            split = max(1, len(adv_iters) // 3)
            all_early.extend(abs(a) for itr in adv_iters[:split]  for a in itr)
            all_late.extend( abs(a) for itr in adv_iters[-split:] for a in itr)

        bins = np.linspace(0, max(max(all_early, default=0.5),
                                   max(all_late,  default=0.5)) + 0.01, 25)
        ax.hist(all_early, bins=bins, alpha=0.6, label="early", density=True)
        ax.hist(all_late,  bins=bins, alpha=0.6, label="late",  density=True)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("|advantage|")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle("|Advantage| distribution: early vs late training (per variant)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANT_ORDER = ["all_seeds", "top_k_4", "top_k_2", "top_k_1"]


def parse_args():
    p = argparse.ArgumentParser(description="Plot ES advantage distributions")
    p.add_argument("--exp-dir", required=True, help="Experiment directory (e.g. results_remote/exp_top_k_n8_<ts>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    # Discover variant dirs in preferred order, fall back to sorted
    variant_dirs = {
        d.name: d
        for d in sorted(exp_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    }
    ordered = [k for k in VARIANT_ORDER if k in variant_dirs]
    ordered += [k for k in variant_dirs if k not in ordered]

    print(f"Found variants: {ordered}")

    data = {}
    for label in ordered:
        d = collect_variant(variant_dirs[label])
        d["_dir"] = variant_dirs[label]
        data[label] = d
        n = len(d["all_abs_flat"])
        print(f"  {label}: {n} advantage values loaded")

    plot_dist(data, exp_dir / "advantage_dist.png")
    plot_cv(data,   exp_dir / "advantage_cv.png")
    plot_early_vs_late(data, exp_dir / "advantage_early_late.png")


if __name__ == "__main__":
    main()
