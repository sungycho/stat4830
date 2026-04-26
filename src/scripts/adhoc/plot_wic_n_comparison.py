"""Plot WIC val_acc curves for N=8 and N=32, stitching consecutive resumed runs.

Usage:
  uv run python -m src.scripts.plot_wic_n_comparison
  uv run python -m src.scripts.plot_wic_n_comparison --out results/wic_comparison.png
"""
from __future__ import annotations

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_logs(*paths: str) -> tuple[list[int], list[float], float | None]:
    all_fwds: list[int] = []
    all_accs: list[float] = []
    baseline = None
    offset = 0

    for path in paths:
        seg_fwds: list[int] = []
        seg_accs: list[float] = []
        for line in Path(path).read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "baseline" and baseline is None:
                baseline = d["val_acc"]
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

    return all_fwds, all_accs, baseline


CURVES: dict[str, list[str]] = {
    "N=8": [
        "results/wic_20260417_193859/log.jsonl",
        "results/wic_20260417_195402/log.jsonl",
    ],
    "N=32": [
        "results/wic_20260417_225319/log.jsonl",
        "results/wic_20260417_231829/log.jsonl",
    ],
    "N=32 (single run)": [
        "results/wic_20260418_001054/log.jsonl",
    ],
}

COLORS = ["steelblue", "tomato", "seagreen"]
STYLES = ["-", "-", "--"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/wic_n_comparison.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, paths), color, ls in zip(CURVES.items(), COLORS, STYLES):
        fwds, accs, baseline = load_logs(*paths)
        if baseline is not None:
            fwds = [0] + fwds
            accs = [baseline] + accs
        ax.plot(fwds, accs, marker="o", linestyle=ls, color=color,
                label=label, linewidth=2, markersize=4)

    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Qwen2.5-1.5B — WIC: population size comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
