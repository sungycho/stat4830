"""Plot val_acc vs training forward passes for seed44 population scaling runs.

Usage:
  uv run python -m src.scripts.plot_seed44_pop_scale
  uv run python -m src.scripts.plot_seed44_pop_scale --dir results/seed44_pop_scale --out results/seed44_pop_scale/pop_scaling_seed44.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path: Path) -> tuple[list[int], list[float], float | None]:
    fwds, accs = [], []
    baseline = None
    for line in path.read_text().splitlines():
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


def parse_n(name: str) -> int | None:
    m = re.search(r"_n(\d+)_", name)
    return int(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/seed44_pop_scale")
    p.add_argument("--out", default="results/seed44_pop_scale/pop_scaling_seed44.png")
    args = p.parse_args()

    root = Path(args.dir)
    runs = sorted(
        [(parse_n(d.name), d) for d in root.iterdir() if d.is_dir() and parse_n(d.name) is not None],
        key=lambda x: x[0],
    )

    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3",
              "#a65628", "#f781bf", "#999999"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for (n, run_dir), color in zip(runs, colors):
        log = run_dir / "log.jsonl"
        if not log.exists():
            continue
        fwds, accs, baseline = load_log(log)
        if not fwds:
            continue
        if baseline is not None:
            fwds = [0] + fwds
            accs = [baseline] + accs
        ax.plot(fwds, accs, marker="o", markersize=3, linewidth=1.8,
                color=color, label=f"N={n}")

    ax.set_xlabel("Training forward passes")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("GSM8K population scaling — Qwen2.5-1.5B (seed 44)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
