"""Plot GSM8K population scaling for N=1, 2 (norm-off), 8, 16 (seeds 42/43/44), cut off at 8000.

Usage:
  uv run python -m src.scripts.plot_pop_scaling_n1_8_16_with_n2
  uv run python -m src.scripts.plot_pop_scaling_n1_8_16_with_n2 --out results/pop_scaling_n1_8_16_with_n2_8k.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID_MAX = 8000
N_PTS = 100

RUNS: dict[str, list[list[str]]] = {
    "N=1": [
        [
            "results/seed42_pop_scale/[1st] gsm8k_n1_20260413_231107/log.jsonl",
            "results/seed42_pop_scale/[2nd] gsm8k_n1_20260414_001549/log.jsonl",
        ],
        ["results/seed43_pop_scale/gsm8k_n1_s2/log.jsonl"],
        ["results/seed44_pop_scale/gsm8k_n1_s3_20260420_215102/log.jsonl"],
    ],
    "N=2 (norm off)": [
        ["results/gsm8k_20260418_022046/gsm8k_20260417_232206/log.jsonl"],  # seed 42
        ["results/gsm8k_n2_norm_off_s43/log.jsonl"],                         # seed 43
        ["results/gsm8k_n2_norm_off_s44/log.jsonl"],                         # seed 44
    ],
    "N=8": [
        ["results/seed42_pop_scale/gsm8k_n8_20260413_021840/log.jsonl"],
        ["results/seed43_pop_scale/gsm8k_n8_s2/log.jsonl"],
        ["results/seed44_pop_scale/gsm8k_n8_s3_20260420_215102/log.jsonl"],
    ],
    "N=16": [
        ["results/seed42_pop_scale/gsm8k_n16_20260412_231214/log.jsonl"],
        ["results/seed43_pop_scale/gsm8k_n16_s2/log.jsonl"],
        ["results/seed44_pop_scale/gsm8k_n16_s3_20260420_215102/log.jsonl"],
    ],
}

COLORS = {"N=1": "#e41a1c", "N=2 (norm off)": "#ff7f00", "N=8": "#377eb8", "N=16": "#984ea3"}


def load_stitched(paths: list[str]) -> tuple[np.ndarray, np.ndarray, float | None]:
    all_fwds: list[int] = []
    all_accs: list[float] = []
    baseline = None

    for path in paths:
        seg_fwds: list[int] = []
        seg_accs: list[float] = []
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
                    seg_fwds.append(x)
                    seg_accs.append(d["val_acc"])
        if seg_fwds:
            last_seen = all_fwds[-1] if all_fwds else 0
            offset = last_seen if seg_fwds[0] <= last_seen else 0
            all_fwds.extend(x + offset for x in seg_fwds)
            all_accs.extend(seg_accs)

    return np.array(all_fwds, dtype=float), np.array(all_accs), baseline


def average_seeds(seed_runs: list[list[str]], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for paths in seed_runs:
        fwds, accs, baseline = load_stitched(paths)
        if fwds.size == 0:
            continue
        if baseline is not None:
            fwds = np.concatenate([[0.0], fwds])
            accs = np.concatenate([[baseline], accs])
        interp = np.interp(grid, fwds, accs, right=np.nan)
        interp[grid > fwds[-1]] = np.nan
        curves.append(interp)

    stacked = np.array(curves)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/pop_scaling_n1_8_16_with_n2_8k.png")
    args = p.parse_args()

    grid = np.linspace(0, GRID_MAX, N_PTS)

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, seed_runs in RUNS.items():
        mean, std = average_seeds(seed_runs, grid)
        color = COLORS[label]
        valid = ~np.isnan(mean)
        ax.plot(grid[valid], mean[valid], color=color, linewidth=2, label=label)
        ax.fill_between(
            grid[valid],
            (mean - std)[valid],
            (mean + std)[valid],
            alpha=0.18, color=color,
        )

    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("GSM8K population scaling — Qwen2.5-1.5B (mean ± std, seeds 42/43/44)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
