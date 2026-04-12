"""P(A=0) degeneracy calculator.

Usage:
  uv run python -m src.utils.pdegen_calc --p0 0.306 --rho 0.38 --B 16
  uv run python -m src.utils.pdegen_calc --p0 0.466 --rho 0.45 --B 32
  uv run python -m src.utils.pdegen_calc --p0 0.3 --rho 0.4 --B 4 8 16 32
"""
from __future__ import annotations

import argparse
import math


def p_degenerate(p0: float, rho: float, B: int) -> tuple[float, str]:
    """P(A=0) with formula selection based on B.

    B=1  : exact  — p0^2 + (1-p0)^2 + 2*p0*(1-p0)*rho
    B>=8 : CLT    — 1/sqrt(4*pi*B*p0*(1-p0)*(1-rho))
    2<=B<8: CLT with warning (approximation is rough)
    """
    if B == 1:
        val = p0**2 + (1 - p0)**2 + 2 * p0 * (1 - p0) * rho
        return min(val, 1.0), "exact"
    denom = 4 * math.pi * B * p0 * (1 - p0) * (1 - rho)
    val = 1.0 / math.sqrt(denom) if denom > 0 else float("nan")
    note = "CLT" if B >= 8 else "CLT~"
    return min(val, 1.0), note


def n_min(p_deg: float, delta: float = 0.05) -> int:
    """Smallest N s.t. P(all N seeds degenerate) <= delta."""
    if p_deg <= 0 or p_deg >= 1:
        return 1
    return math.ceil(math.log(delta) / math.log(p_deg))


def main():
    p = argparse.ArgumentParser(description="P(A=0) degeneracy calculator")
    p.add_argument("--p0",    type=float, required=True,  help="Base accuracy p0 in (0,1)")
    p.add_argument("--rho",   type=float, required=True,  help="Intra-pair correlation rho in [0,1)")
    p.add_argument("--B",     type=int,   nargs="+", required=True, help="Batch size(s)")
    p.add_argument("--delta", type=float, default=0.05,   help="Failure probability for Nmin (default 0.05)")
    args = p.parse_args()

    print(f"\np0={args.p0}  rho={args.rho}  delta={args.delta}\n")
    print(f"{'B':>4}  {'P(A=0)':>8}  {'Nmin':>5}  {'formula':>8}")
    print("-" * 32)
    for B in args.B:
        pd, note = p_degenerate(args.p0, args.rho, B)
        nm = n_min(pd, args.delta)
        print(f"{B:>4}  {pd:>8.4f}  {nm:>5}  {note:>8}")
    print("  CLT~: approximation rough (B<8), exact used at B=1\n")


if __name__ == "__main__":
    main()
