"""Plot val_acc vs cumulative forward passes from experiment logs.

Usage:
  # Single block — reads results/exp_one_vs_two_<ts>/
  uv run python -m src.scripts.plot_results --exp-dir results/exp_one_vs_two_20240301_120000

  # Compare variants inside a block dir (subdirs are variant names)
  uv run python -m src.scripts.plot_results --exp-dir results/exp_one_vs_two_20240301_120000 --out fig.png

  # Compare multiple blocks
  uv run python -m src.scripts.plot_results --exp-dir results/exp_all_20240301/ --recursive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path: Path) -> list[dict]:
    entries = []
    for line in log_path.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def extract_curve(entries: list[dict]) -> tuple[list[int], list[float]]:
    """Extract (train_fwd, val_acc) pairs from log entries.

    Uses 'train_fwd' (training passes only, the budget-controlled axis).
    Falls back to 'cumulative_fwd' for logs produced before the split was introduced.
    """
    fwds, accs = [], []
    for e in entries:
        if e.get("event") == "iter" and "val_acc" in e:
            x = e.get("train_fwd") or e.get("cumulative_fwd")
            if x is not None:
                fwds.append(x)
                accs.append(e["val_acc"])
    return fwds, accs


def find_seed_logs(variant_dir: Path) -> list[Path]:
    """Find all log.jsonl files inside seed* subdirs of a variant dir."""
    logs = sorted(variant_dir.glob("seed*/log.jsonl"))
    if not logs:
        # Maybe logs are directly inside variant_dir (single run)
        direct = variant_dir / "log.jsonl"
        if direct.exists():
            logs = [direct]
    return logs


def collect_variant_curves(variant_dir: Path) -> tuple[list[int], list[float], list[float]]:
    """Load all seeds for a variant, return (fwds, mean_accs, std_accs) on a common grid.

    Seeds may have different x-positions (fwd counts) if they diverge in length.
    We interpolate each curve onto the intersection of all seeds' fwd ranges so
    every point on the shared grid has a valid value from every seed.
    """
    seed_curves = []
    for log_path in find_seed_logs(variant_dir):
        entries = load_log(log_path)
        fwds, accs = extract_curve(entries)
        if fwds:
            seed_curves.append((np.array(fwds, dtype=float), np.array(accs)))

    if not seed_curves:
        return [], [], []

    if len(seed_curves) == 1:
        f, a = seed_curves[0]
        return f.tolist(), a.tolist(), [0.0] * len(a)

    # Common grid: from max(first_x) to min(last_x), 100 evenly-spaced points
    grid_start = max(c[0][0] for c in seed_curves)
    grid_end   = min(c[0][-1] for c in seed_curves)
    if grid_end <= grid_start:
        # No overlap — fall back to shortest truncation
        min_len = min(len(c[0]) for c in seed_curves)
        fwds = seed_curves[0][0][:min_len]
        stacked = np.array([c[1][:min_len] for c in seed_curves])
        return fwds.tolist(), stacked.mean(axis=0).tolist(), stacked.std(axis=0).tolist()

    n_points = min(100, min(len(c[0]) for c in seed_curves))
    grid = np.linspace(grid_start, grid_end, n_points)
    interpolated = np.array([np.interp(grid, c[0], c[1]) for c in seed_curves])
    return grid.tolist(), interpolated.mean(axis=0).tolist(), interpolated.std(axis=0).tolist()


def plot_block(exp_dir: Path, out_path: Path, title: str | None = None) -> None:
    """Plot all variants inside exp_dir as separate lines."""
    variant_dirs = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for vdir in variant_dirs:
        fwds, means, stds = collect_variant_curves(vdir)
        if not fwds:
            continue
        fwds_k = [f / 1000 for f in fwds]
        line, = ax.plot(fwds_k, means, marker="o", markersize=3, label=vdir.name)
        stds_arr = np.array(stds)
        means_arr = np.array(means)
        ax.fill_between(fwds_k, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.2, color=line.get_color())
        plotted += 1

    if plotted == 0:
        print(f"[warn] No val curves found in {exp_dir}")
        plt.close(fig)
        return

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(title or exp_dir.name)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_calibration_heatmap(exp_dir: Path, out_path: Path) -> None:
    """Special plot for calibration block: heatmap of best_val over sigma × lr grid.

    Cell text is "mean±std" when std exists (n_seeds >= 2), else just mean.
    """
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        print(f"[warn] No summary.json in {exp_dir}, skipping heatmap")
        return

    with open(summary_path) as f:
        results = json.load(f)

    sigma_vals = sorted({r["config"].get("sigma") for r in results if "sigma" in r.get("config", {})})
    lr_vals = sorted({r["config"].get("lr") for r in results if "lr" in r.get("config", {})})

    if not sigma_vals or not lr_vals:
        return

    mean_grid = np.full((len(sigma_vals), len(lr_vals)), float("nan"))
    std_grid = np.full((len(sigma_vals), len(lr_vals)), float("nan"))
    for r in results:
        cfg = r.get("config", {})
        if "sigma" not in cfg or "lr" not in cfg:
            continue
        i = sigma_vals.index(cfg["sigma"])
        j = lr_vals.index(cfg["lr"])
        if r.get("mean_best_val") is not None:
            mean_grid[i, j] = r["mean_best_val"]
        if r.get("std_best_val") is not None:
            std_grid[i, j] = r["std_best_val"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mean_grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean best val_acc")
    ax.set_xticks(range(len(lr_vals)))
    ax.set_xticklabels([f"{v:.0e}" for v in lr_vals])
    ax.set_yticks(range(len(sigma_vals)))
    ax.set_yticklabels([f"{v:.0e}" for v in sigma_vals])
    ax.set_xlabel("lr (alpha)")
    ax.set_ylabel("sigma")
    ax.set_title("Calibration: val_acc heatmap (sigma × lr)")

    for i in range(len(sigma_vals)):
        for j in range(len(lr_vals)):
            mv = mean_grid[i, j]
            sv = std_grid[i, j]
            if np.isnan(mv):
                text = "N/A"
            elif np.isnan(sv):
                text = f"{mv:.3f}"
            else:
                text = f"{mv:.3f}\n±{sv:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot ES experiment results")
    p.add_argument("--exp-dir",  required=True, help="Experiment output directory")
    p.add_argument("--out",      default=None,  help="Output figure path (default: <exp-dir>/fig.png)")
    p.add_argument("--title",    default=None,  help="Plot title override")
    p.add_argument("--recursive", action="store_true",
                   help="Recursively plot each block subdirectory")
    p.add_argument("--heatmap",  action="store_true",
                   help="Also produce calibration heatmap if summary.json exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    if args.recursive:
        # Each subdirectory is a block
        for block_dir in sorted(d for d in exp_dir.iterdir() if d.is_dir()):
            out = block_dir / "fig.png"
            plot_block(block_dir, out, title=block_dir.name)
            if args.heatmap:
                plot_calibration_heatmap(block_dir, block_dir / "calibration_heatmap.png")
    else:
        out = Path(args.out) if args.out else exp_dir / "fig.png"
        plot_block(exp_dir, out, title=args.title)
        if args.heatmap:
            plot_calibration_heatmap(exp_dir, exp_dir / "calibration_heatmap.png")


if __name__ == "__main__":
    main()
