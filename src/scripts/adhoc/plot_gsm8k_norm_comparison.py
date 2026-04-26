"""Plot GSM8K N=2 val_acc: normalization on vs off, averaged across 3 seeds.

Usage:
  uv run python -m src.scripts.plot_gsm8k_norm_comparison
  uv run python -m src.scripts.plot_gsm8k_norm_comparison --out results/gsm8k_norm_comparison.png
"""
from __future__ import annotations

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log(path: str) -> tuple[list[int], list[float], float | None]:
    fwds, accs = [], []
    baseline = None
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "baseline" and baseline is None:
            baseline = d.get("val_acc")
        if d.get("event") == "iter" and "val_acc" in d:
            x = d.get("train_fwd") or d.get("cumulative_fwd")
            if x is not None:
                fwds.append(x)
                accs.append(d["val_acc"])
    return fwds, accs, baseline


SEEDS: dict[str, list[str]] = {
    "normalize=on": [
        "results/gsm8k_20260418_022046/gsm8k_20260417_232206/log.jsonl",  # s42, norm OFF -> wait
        # corrected below
    ],
}

# seed log paths per condition
NORM_OFF_LOGS = [
    "results/gsm8k_20260418_022046/gsm8k_20260417_232206/log.jsonl",  # seed 42
    "results/gsm8k_n2_norm_off_s43/log.jsonl",                         # seed 43
    "results/gsm8k_n2_norm_off_s44/log.jsonl",                         # seed 44
]

NORM_ON_LOGS = [
    "results/gsm8k_n2_20260414_010958/log.jsonl",  # seed 42 (short, up to 6400)
    "results/gsm8k_n2_norm_on_s43/log.jsonl",       # seed 43
    "results/gsm8k_n2_norm_on_s44/log.jsonl",       # seed 44
]

GRID_MAX = 12800
N_PTS = 100


def average_seeds(log_paths: list[str], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each seed onto grid; seeds that end early get NaN beyond their last point."""
    curves = []
    for path in log_paths:
        fwds, accs, baseline = load_log(path)
        if not fwds:
            continue
        if baseline is not None:
            fwds = [0] + fwds
            accs = [baseline] + accs
        fwds_a = np.array(fwds, dtype=float)
        accs_a = np.array(accs, dtype=float)
        interp = np.interp(grid, fwds_a, accs_a, right=np.nan)
        # NaN beyond last observed point
        interp[grid > fwds_a[-1]] = np.nan
        curves.append(interp)

    stacked = np.array(curves)  # (n_seeds, n_pts)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gsm8k_norm_comparison.png")
    args = p.parse_args()

    grid = np.linspace(0, GRID_MAX, N_PTS)

    conditions = [
        ("normalize=off (3 seeds)", NORM_OFF_LOGS, "tomato"),
        ("normalize=on (3 seeds)",  NORM_ON_LOGS,  "steelblue"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, logs, color in conditions:
        mean, std = average_seeds(logs, grid)
        ax.plot(grid, mean, color=color, linewidth=2, label=label)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Qwen2.5-1.5B — GSM8K N=2: normalize on vs off (mean ± std, seeds 42/43/44)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
