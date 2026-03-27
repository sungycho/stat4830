"""Plot LOZO medium-suite outputs from summary/manifest artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_RE = re.compile(
    r"([A-Za-z0-9_./-]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)


def _choose_metric(
    metrics: dict[str, float],
) -> tuple[str | None, float | None]:
    if not metrics:
        return None, None
    ordered = sorted(metrics.items())
    for key, value in ordered:
        lower = key.lower()
        if "acc" in lower or "f1" in lower or "mcc" in lower:
            return key, value
    return ordered[0]


def _extract_series(log_path: Path) -> tuple[list[int], list[float]]:
    if not log_path.exists():
        return [], []
    series: list[float] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "eval" not in line.lower():
            continue
        for key, raw_val in METRIC_RE.findall(line):
            if "acc" not in key.lower() and "loss" not in key.lower():
                continue
            try:
                series.append(float(raw_val))
            except ValueError:
                continue
    if not series:
        return [], []
    return list(range(1, len(series) + 1)), series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create LOZO-only plots from suite summary.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--summary",
        required=True,
        help="Path to summary.json from a suite run",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory for figures "
            "(default: sibling `figures/` next to summary)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    payload = json.loads(summary_path.read_text())
    records = payload.get("records", [])

    lozo_records = [
        r
        for r in records
        if r.get("method") == "lozo"
        and r.get("status") in {"completed", "skipped"}
    ]
    if not lozo_records:
        raise RuntimeError("No LOZO completed/skipped records found in summary.")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else summary_path.parent / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: final metric by task (mean across seeds)
    bucket: dict[tuple[str, int], list[float]] = defaultdict(list)
    for rec in lozo_records:
        metric_name, metric_val = _choose_metric(rec.get("metrics", {}))
        if metric_val is None:
            continue
        task = str(rec.get("task"))
        k = int(rec.get("k"))
        bucket[(task, k)].append(float(metric_val))

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for (task, k), vals in sorted(bucket.items()):
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        labels.append(f"{task} (k={k})")
        means.append(m)
        stds.append(v**0.5)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Final metric")
    ax.set_title("LOZO final metric by task")
    fig.tight_layout()
    metric_out = out_dir / "lozo_final_metric_by_task.png"
    fig.savefig(metric_out, dpi=180)
    plt.close(fig)

    # Plot 2: peak GPU memory usage by run
    mem_labels: list[str] = []
    mem_values: list[float] = []
    for rec in lozo_records:
        peak = rec.get("gpu_memory_mb_peak")
        if peak is None:
            continue
        mem_labels.append(
            f"{rec.get('task')}-k{rec.get('k')}-s{rec.get('seed')}"
        )
        mem_values.append(float(peak))

    if mem_values:
        fig2, ax2 = plt.subplots(figsize=(max(9, len(mem_values) * 0.7), 5))
        x2 = list(range(len(mem_values)))
        ax2.plot(x2, mem_values, marker="o")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(mem_labels, rotation=45, ha="right")
        ax2.set_ylabel("Peak GPU memory (MB)")
        ax2.set_title("LOZO peak GPU memory by run")
        fig2.tight_layout()
        mem_out = out_dir / "lozo_peak_gpu_memory.png"
        fig2.savefig(mem_out, dpi=180)
        plt.close(fig2)

    # Plot 3: first available log-derived eval trajectory
    trajectory_written = False
    for rec in lozo_records:
        log_path = Path(str(rec.get("log_path", "")))
        xs, ys = _extract_series(log_path)
        if not xs:
            continue
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        ax3.plot(xs, ys, linewidth=2)
        ax3.set_xlabel("Logged eval index")
        ax3.set_ylabel("Metric value")
        ax3.set_title(
            "LOZO eval trajectory: "
            f"{rec.get('task')} k={rec.get('k')} seed={rec.get('seed')}"
        )
        fig3.tight_layout()
        traj_out = out_dir / "lozo_eval_trajectory_example.png"
        fig3.savefig(traj_out, dpi=180)
        plt.close(fig3)
        trajectory_written = True
        break

    print("Wrote figures to:", out_dir)
    print("- lozo_final_metric_by_task.png")
    if mem_values:
        print("- lozo_peak_gpu_memory.png")
    if trajectory_written:
        print("- lozo_eval_trajectory_example.png")


if __name__ == "__main__":
    main()
