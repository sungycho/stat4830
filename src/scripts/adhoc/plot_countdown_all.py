"""Plot all Countdown ES results: N=1,2,4,8,16,32,64.

Handles the mixed directory structures from separate transfers.

Usage:
  uv run python -m src.scripts.plot_countdown_all \
      --base-dir results_remote \
      --out      results_remote/countdown_all.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "N1":       "#a6cee3",
    "N2":       "#1f78b4",
    "N4":       "#b2df8a",
    "N4_nonorm":"#33a02c",
    "N8":       "#fb9a99",
    "N16":      "#e31a1c",
    "N32":      "#fdbf6f",
    "N64":      "#ff7f00",
}

# (label, exp_dir_glob, seed_glob_within_exp)
# seed_glob: relative to the matched exp dir
VARIANTS = [
    ("N1",        "exp_countdown_N1_*",   "seed*/log.jsonl"),
    ("N2",        "exp_countdown_N2_*",   "seed*/log.jsonl"),
    ("N4",        "exp_countdown_N4_*",   "seed*/log.jsonl"),
    ("N4_nonorm", "exp_countdown_N4_*",   "N4_nonorm/seed*/log.jsonl"),
    ("N8",        "exp_countdown_N8_*",   "N8/seed*/log.jsonl"),
    ("N16",       "exp_countdown_N16_*",  "seed*/log.jsonl"),
    ("N32",       "exp_countdown_N32_*",  "N32/seed*/log.jsonl"),
    ("N64",       "exp_countdown_N64_*",  "N64/seed*/log.jsonl"),
]


def load_logs(base_dir: Path, exp_glob: str, seed_glob: str):
    """Find all matching logs and return per-x val_acc lists."""
    exp_dirs = sorted(base_dir.glob(exp_glob))
    if not exp_dirs:
        return None, None, None

    exp_dir = exp_dirs[0]   # take the first match
    logs = sorted(exp_dir.glob(seed_glob))
    if not logs:
        return None, None, None

    all_curves: dict[int, list[float]] = {}
    baselines = []

    for log in logs:
        baseline = None
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "baseline":
                baseline = e["val_acc"]
            if e.get("event") == "iter" and "val_acc" in e:
                x = e.get("train_fwd") or e.get("cumulative_fwd", 0)
                all_curves.setdefault(x, []).append(e["val_acc"])
        if baseline is not None:
            baselines.append(baseline)

    if not all_curves:
        return None, None, None

    baseline_mean = float(np.mean(baselines)) if baselines else 0.0
    xs = sorted(all_curves)
    means = [float(np.mean(all_curves[x])) for x in xs]
    stds  = [float(np.std(all_curves[x]))  for x in xs]

    fwds  = np.array([0.0] + [x / 1000 for x in xs])
    means = np.array([baseline_mean] + means)
    stds  = np.array([0.0] + stds)
    return fwds, means, stds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="results_remote")
    p.add_argument("--out", default="results_remote/countdown_all.png")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, exp_glob, seed_glob in VARIANTS:
        fwds, means, stds = load_logs(base, exp_glob, seed_glob)
        if fwds is None:
            print(f"[skip] {label}: no data found")
            continue

        color = COLORS.get(label, "#888888")
        ls = "--" if "nonorm" in label.lower() else "-"
        ax.plot(fwds, means, marker="o", markersize=3, linewidth=1.8,
                color=color, linestyle=ls, label=label)
        ax.fill_between(fwds, means - stds, means + stds,
                        alpha=0.12, color=color)
        print(f"{label}: {len(fwds)-1} val points, peak={max(means):.4f}")

    ax.set_xlabel("Training forward passes (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 0.15)
    ax.set_title("Countdown — Qwen2.5-3B-Instruct ES\nPop-size scaling (N=1→64)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
