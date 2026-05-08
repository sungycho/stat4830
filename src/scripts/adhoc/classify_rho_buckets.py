"""Classify every (model, task) entry into a 3×3 regime grid.

Axes:
  ρ  (per-example rho) : low < RHO_LOW  |  RHO_LOW ≤ moderate < RHO_HIGH  |  high ≥ RHO_HIGH
  p₀ (base accuracy)   : low < P0_LOW   |  P0_LOW  ≤ moderate < P0_HIGH   |  high ≥ P0_HIGH

Edit the four constants below to change the bucket boundaries.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
RHO_LOW  = 0.3   # ρ < RHO_LOW  → low;  RHO_LOW ≤ ρ < RHO_HIGH → moderate;  ρ ≥ RHO_HIGH → high
RHO_HIGH = 0.7

P0_LOW   = 0.2   # p₀ < P0_LOW  → low;  P0_LOW ≤ p₀ < P0_HIGH → moderate;  p₀ ≥ P0_HIGH → high
P0_HIGH  = 0.8

CSV_DIR  = "results"   # directory containing rho_sweep_rho.csv and rho_sweep_baseline.csv
# ---------------------------------------------------------------------------

RHO_LABELS = ["low", "moderate", "high"]
P0_LABELS  = ["low", "moderate", "high"]


def _band(val: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    if val < lo:
        return "low"
    if val < hi:
        return "moderate"
    return "high"


def load_wide_csv(path: Path) -> dict[tuple[str, str], float]:
    """Return {(model, task): value} from a wide-format CSV (tasks as rows, models as columns)."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            for model, cell in row.items():
                if model == "task" or not cell.strip():
                    continue
                try:
                    out[(model, task)] = float(cell)
                except ValueError:
                    pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir",        default=CSV_DIR)
    p.add_argument("--rho-thresholds", type=float, nargs=2, default=[RHO_LOW, RHO_HIGH],
                   metavar=("LO", "HI"))
    p.add_argument("--p0-thresholds",  type=float, nargs=2, default=[P0_LOW, P0_HIGH],
                   metavar=("LO", "HI"))
    args = p.parse_args()

    csv_dir = Path(args.csv_dir)
    rho_path = csv_dir / "rho_sweep_rho.csv"
    p0_path  = csv_dir / "rho_sweep_baseline.csv"

    for path in (rho_path, p0_path):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

    rho_data = load_wide_csv(rho_path)
    p0_data  = load_wide_csv(p0_path)

    rho_t = tuple(args.rho_thresholds)
    p0_t  = tuple(args.p0_thresholds)

    # Build buckets: {(rho_band, p0_band): [(model, task, rho, p0)]}
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    skipped = []

    all_keys = set(rho_data) | set(p0_data)
    for model, task in sorted(all_keys):
        rho = rho_data.get((model, task))
        p0  = p0_data.get((model, task))
        if rho is None or p0 is None or math.isnan(rho) or math.isnan(p0):
            skipped.append((model, task, rho, p0))
            continue
        rb = _band(rho, rho_t)
        pb = _band(p0, p0_t)
        buckets[(rb, pb)].append((model, task, rho, p0))

    # Print
    rho_lo, rho_hi = args.rho_thresholds
    p0_lo,  p0_hi  = args.p0_thresholds
    print(f"Thresholds — ρ: low<{rho_lo} | moderate<{rho_hi} | high≥{rho_hi}")
    print(f"           — p₀: low<{p0_lo}  | moderate<{p0_hi}  | high≥{p0_hi}")
    print()

    total = 0
    for p0_band in P0_LABELS:
        for rho_band in RHO_LABELS:
            entries = buckets.get((rho_band, p0_band), [])
            total += len(entries)
            print(f"┌─ ρ={rho_band:8s}  p₀={p0_band:8s}  ({len(entries)} entries)")
            for model, task, rho, p0 in entries:
                print(f"│    {model:<30s}  {task:<12s}  ρ={rho:+.3f}  p₀={p0:.3f}")
            print()

    print(f"Total classified: {total}  |  Skipped (missing/nan): {len(skipped)}")
    if skipped:
        print("Skipped entries:")
        for model, task, rho, p0 in skipped:
            print(f"  {model}  {task}  rho={rho}  p0={p0}")

    # Summary count table
    print()
    print("Count table (rows=p₀, cols=ρ):")
    col_w = 10
    print(f"{'':12s}" + "".join(f"{'ρ='+b:>{col_w}}" for b in RHO_LABELS))
    for pb in P0_LABELS:
        row = f"{'p₀='+pb:<12s}"
        for rb in RHO_LABELS:
            n = len(buckets.get((rb, pb), []))
            row += f"{n:>{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
