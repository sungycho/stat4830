"""Plot Qwen2.5-1.5B val_acc on WIC (N=32, stitched from two consecutive runs).

Usage:
  uv run python -m src.scripts.plot_wic_qwen
  uv run python -m src.scripts.plot_wic_qwen --out results/wic_qwen.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = [
    "results/wic_20260417_225319/log.jsonl",
    "results/wic_20260417_231829/log.jsonl",
]


def load_stitched(paths: list[str]) -> tuple[list[int], list[float], float | None]:
    all_fwds: list[int] = []
    all_accs: list[float] = []
    baseline = None

    for path in paths:
        seg_fwds, seg_accs = [], []
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

    return all_fwds, all_accs, baseline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/wic_qwen.png")
    args = p.parse_args()

    fwds, accs, baseline = load_stitched(LOGS)
    if baseline is not None:
        fwds = [0] + fwds
        accs = [baseline] + accs

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fwds, accs, color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random baseline (0.5)")
    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Qwen2.5-1.5B — WIC (N=32, normalize=on, seed=42)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
