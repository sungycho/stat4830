"""Plot LR sweep results for MNLI ES fine-tuning.

Reads log.jsonl files from a sweep directory and plots val_acc vs train_fwd
for each learning rate.

Usage:
  uv run python -m src.scripts.adhoc.plot_lr_sweep_mnli results/mnli_lr_sweep_20260503_191049
  uv run python -m src.scripts.adhoc.plot_lr_sweep_mnli results/mnli_lr_sweep_20260503_191049 --out plots/mnli_lr_sweep.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_run(log_path: Path) -> tuple[float | None, list[int], list[float]]:
    """Return (baseline_acc, train_fwd_steps, val_accs) from a log.jsonl."""
    baseline_acc = None
    train_fwds: list[int] = []
    val_accs: list[float] = []

    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "baseline":
            baseline_acc = entry.get("val_acc")
        if entry.get("event") == "iter" and "val_acc" in entry:
            train_fwds.append(entry["train_fwd"])
            val_accs.append(entry["val_acc"])

    return baseline_acc, train_fwds, val_accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", help="Path to sweep directory (contains lr_* subdirs)")
    p.add_argument("--out", default=None, help="Output path (default: <sweep_dir>/lr_sweep_plot.png)")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else sweep_dir / "lr_sweep_plot.png"

    # Collect all lr_* subdirs that have a log.jsonl
    runs = []
    for d in sorted(sweep_dir.iterdir()):
        log = d / "log.jsonl"
        if d.is_dir() and log.exists() and d.name.startswith("lr_"):
            # Parse LR from dirname: lr_1e-05 → 1e-5
            try:
                lr = float(d.name.replace("lr_", ""))
            except ValueError:
                continue
            baseline, fwds, accs = load_run(log)
            if fwds:
                runs.append((lr, baseline, fwds, accs))

    if not runs:
        raise SystemExit(f"No valid lr_* runs found in {sweep_dir}")

    runs.sort(key=lambda x: x[0], reverse=True)  # highest LR first

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = plt.cm.viridis([i / (len(runs) - 1) for i in range(len(runs))]) if len(runs) > 1 else ["steelblue"]

    baseline_shown = False
    for (lr, baseline, fwds, accs), color in zip(runs, colors):
        label = f"lr={lr:.0e}"
        ax.plot(fwds, accs, marker="o", markersize=4, linewidth=1.8,
                label=label, color=color)
        if baseline is not None and not baseline_shown:
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                       label=f"baseline ({baseline:.3f})")
            baseline_shown = True

    # Random chance line for 3-class MNLI
    ax.axhline(1/3, color="lightcoral", linestyle=":", linewidth=1, label="random (0.333)")

    ax.set_xlabel("Train forward passes")
    ax.set_ylabel("Val accuracy")
    ax.set_title("LR sweep — Qwen2.5-1.5B-Instruct × MNLI (ES, accuracy reward)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")

    # Print final val_acc table
    print(f"\n{'LR':>10s}  {'baseline':>9s}  {'best_val':>9s}  {'final_val':>9s}  {'delta':>8s}")
    for lr, baseline, fwds, accs in runs:
        best = max(accs)
        final = accs[-1]
        delta = best - baseline if baseline is not None else float("nan")
        b = f"{baseline:.3f}" if baseline is not None else "  n/a"
        print(f"{lr:>10.1e}  {b:>9s}  {best:>9.3f}  {final:>9.3f}  {delta:>+8.3f}")


if __name__ == "__main__":
    main()
