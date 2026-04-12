"""Plot degeneracy probe results across model sizes and batch sizes.

Usage:
  uv run python -m src.scripts.plot_probe_grid
  uv run python -m src.scripts.plot_probe_grid --results-dir results/degen_probe --output results/degen_probe/probe_grid.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Model display config
# ---------------------------------------------------------------------------
MODEL_ORDER  = ["0.5B", "1.5B", "7B"]
MODEL_LABELS = {"0.5B": "Qwen2.5-0.5B", "1.5B": "Qwen2.5-1.5B", "7B": "Qwen2.5-7B"}
MODEL_COLORS = {"0.5B": "#4C72B0", "1.5B": "#DD8452", "7B": "#55A868"}
MODEL_MARKERS= {"0.5B": "o",        "1.5B": "s",        "7B": "^"}
BATCH_SIZES  = [4, 8, 16, 32]

NAME_TO_KEY = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5B",
    "Qwen/Qwen2.5-7B-Instruct":   "7B",
}


def load_results(results_dir: Path) -> dict:
    """Load all probe JSONs into nested dict [model_key][batch_size] -> data."""
    data: dict[str, dict[int, dict]] = {k: {} for k in MODEL_ORDER}
    for f in sorted(results_dir.glob("probe_gsm8k_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        model_key = NAME_TO_KEY.get(d["model"])
        if model_key is None:
            continue
        B = d["batch_size"]
        # Keep the latest file if duplicates exist
        if B not in data[model_key] or f.stat().st_mtime > data[model_key][B]["_mtime"]:
            d["_mtime"] = f.stat().st_mtime
            data[model_key][B] = d
    return data


def theory_p0(p0: float, rho: float, B: int) -> float:
    denom = 4 * math.pi * B * p0 * (1 - p0) * (1 - rho)
    return 1.0 / math.sqrt(denom) if denom > 0 else float("nan")


def make_plot(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Degeneracy Probe: Qwen2.5 on GSM8K  (σ=0.001, K=200)",
                 fontsize=14, fontweight="bold", y=0.98)

    ax_emp   = axes[0, 0]   # empirical vs theory P(A=0) vs B
    ax_ratio = axes[0, 1]   # theory / empirical ratio vs B
    ax_p0    = axes[1, 0]   # p0 vs B
    ax_rho   = axes[1, 1]   # rho_per_example vs B

    def set_log2_xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.set_xlabel("Batch size B (log₂ scale)", fontsize=11)

    # ---- axes[0,0]: empirical and theory P(A=0) vs B ----
    for mk in MODEL_ORDER:
        Bs, emp, thy = [], [], []
        for B in BATCH_SIZES:
            if B not in data[mk]:
                continue
            d = data[mk][B]
            Bs.append(B)
            emp.append(d["p_degenerate_empirical"])
            thy.append(d["p_degenerate_theory_rho"])
        if not Bs:
            continue
        c = MODEL_COLORS[mk]
        m = MODEL_MARKERS[mk]
        ax_emp.plot(Bs, emp, color=c, marker=m, linewidth=2, markersize=7,
                    label=f"{MODEL_LABELS[mk]} empirical")
        ax_emp.plot(Bs, thy, color=c, marker=m, linewidth=1.5, markersize=7,
                    linestyle="--", alpha=0.6, label=f"{MODEL_LABELS[mk]} theory (ρ_emp)")

    set_log2_xaxis(ax_emp)
    ax_emp.set_ylabel("P(A=0)", fontsize=11)
    ax_emp.set_title("Empirical vs Theory P(A=0)", fontsize=12)
    ax_emp.set_ylim(bottom=0)
    ax_emp.legend(fontsize=7.5, ncol=2)
    ax_emp.grid(True, alpha=0.3, which="both")

    # ---- axes[0,1]: theory/empirical ratio vs B ----
    for mk in MODEL_ORDER:
        Bs, ratios = [], []
        for B in BATCH_SIZES:
            if B not in data[mk]:
                continue
            d = data[mk][B]
            emp = d["p_degenerate_empirical"]
            thy = d["p_degenerate_theory_rho"]
            if emp > 0:
                ratios.append(thy / emp)
                Bs.append(B)
        if not Bs:
            continue
        ax_ratio.plot(Bs, ratios, color=MODEL_COLORS[mk], marker=MODEL_MARKERS[mk],
                      linewidth=2, markersize=7, label=MODEL_LABELS[mk])

    ax_ratio.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="perfect (ratio=1)")
    set_log2_xaxis(ax_ratio)
    ax_ratio.set_ylabel("Theory / Empirical", fontsize=11)
    ax_ratio.set_title("Formula Accuracy (theory / empirical P(A=0))", fontsize=12)
    ax_ratio.legend(fontsize=9)
    ax_ratio.grid(True, alpha=0.3, which="both")

    # ---- axes[1,0]: p0 vs B ----
    for mk in MODEL_ORDER:
        Bs, ys = [], []
        for B in BATCH_SIZES:
            if B not in data[mk]:
                continue
            Bs.append(B)
            ys.append(data[mk][B]["p0_empirical"])
        if Bs:
            ax_p0.plot(Bs, ys, color=MODEL_COLORS[mk], marker=MODEL_MARKERS[mk],
                       linewidth=2, markersize=7, label=MODEL_LABELS[mk])

    set_log2_xaxis(ax_p0)
    ax_p0.set_ylabel("p₀ (base accuracy)", fontsize=11)
    ax_p0.set_title("Base Accuracy p₀ by Batch Size", fontsize=12)
    ax_p0.legend(fontsize=9)
    ax_p0.grid(True, alpha=0.3, which="both")
    ax_p0.set_ylim(0, 1)

    # ---- axes[1,1]: rho_per_example vs B ----
    for mk in MODEL_ORDER:
        Bs, ys = [], []
        for B in BATCH_SIZES:
            if B not in data[mk]:
                continue
            Bs.append(B)
            ys.append(data[mk][B]["rho_per_example"])
        if Bs:
            ax_rho.plot(Bs, ys, color=MODEL_COLORS[mk], marker=MODEL_MARKERS[mk],
                        linewidth=2, markersize=7, label=MODEL_LABELS[mk])

    set_log2_xaxis(ax_rho)
    ax_rho.set_ylabel("ρ (per-example, intra-pair correlation)", fontsize=11)
    ax_rho.set_title("ρ per-example by Batch Size", fontsize=12)
    ax_rho.legend(fontsize=9)
    ax_rho.grid(True, alpha=0.3, which="both")
    ax_rho.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot] saved to {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/degen_probe")
    p.add_argument("--output", default="results/degen_probe/probe_grid.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_results(Path(args.results_dir))
    make_plot(data, Path(args.output))
