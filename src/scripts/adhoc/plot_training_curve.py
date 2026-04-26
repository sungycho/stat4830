"""Plot training curve from a log.jsonl file.

Usage:
  uv run python -m src.scripts.plot_training_curve results/gsm8k_20260412_231214/log.jsonl
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path


def load_log(path: str):
    iters, mean_advs = [], []
    val_iters, val_accs = [], []
    baseline_acc = None

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "event" not in d:
                continue
            if d["event"] == "baseline":
                baseline_acc = d["val_acc"]
            elif d["event"] == "iter":
                iters.append(d["iteration"])
                mean_advs.append(d["mean_adv"])
                if "val_acc" in d:
                    val_iters.append(d["iteration"])
                    val_accs.append(d["val_acc"])

    return iters, mean_advs, val_iters, val_accs, baseline_acc


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/gsm8k_20260412_231214/log.jsonl"
    out_path = Path(path).parent / "training_curve.png"

    iters, mean_advs, val_iters, val_accs, baseline_acc = load_log(path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- top: val accuracy ---
    if baseline_acc is not None:
        ax1.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1, label=f"Baseline {baseline_acc:.3f}")
    ax1.plot(val_iters, val_accs, "o-", color="steelblue", linewidth=2, markersize=6, label="Val acc")
    ax1.set_ylabel("Val Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("ES Training — Qwen2.5-1.5B-Instruct on GSM8K")

    # --- bottom: mean advantage per iter ---
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.bar(iters, mean_advs, color=["steelblue" if a >= 0 else "tomato" for a in mean_advs], alpha=0.7)
    ax2.set_ylabel("Mean Advantage")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
