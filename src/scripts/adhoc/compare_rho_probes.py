"""Compare two degeneracy probe JSON results and print a side-by-side summary.

Reads two probe output files (produced by probe_degeneracy.py) and prints:
  - side-by-side key metrics (p0, rho, P(A=0), N_min)
  - a verdict: which setup shows higher rho (incoherent failure mode indicator)
  - whether the result is consistent with the theory prediction

Usage:
  uv run python -m src.scripts.compare_rho_probes \\
      --file-a results/degen_probe/rho_malignant_llama1b_mnli.json --label-a "LLaMA-1B / MNLI" \\
      --file-b results/degen_probe/rho_malignant_qwen15b_wic.json  --label-b "Qwen-1.5B / WIC"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def nmin_str(r: dict, alpha: float = 0.05) -> str:
    v = r["nmin_table"].get(str(alpha))
    if v is None:
        return "n/a"
    return str(v) if v != float("inf") else "∞"


def fmt(x, decimals: int = 3) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "  NaN "
    return f"{x:.{decimals}f}"


def print_comparison(a: dict, la: str, b: dict, lb: str) -> None:
    col = 28  # label column width
    val = 14  # value column width

    sep = "=" * (col + val * 2 + 4)

    print(sep)
    print("  ρ-PROBE COMPARISON REPORT")
    print(sep)
    print(f"  {'':>{col}}  {la:>{val}}  {lb:>{val}}")
    print("-" * (col + val * 2 + 4))

    rows = [
        ("Model",                   a["model"].split("/")[-1],       b["model"].split("/")[-1]),
        ("Task",                    a["task"],                        b["task"]),
        ("σ (perturbation scale)",  fmt(a["sigma"]),                  fmt(b["sigma"])),
        ("K (probe pairs)",         str(a["K"]),                      str(b["K"])),
        ("B (batch size)",          str(a["batch_size"]),             str(b["batch_size"])),
        ("",                        "",                               ""),
        ("p₀  (base accuracy)",     fmt(a["p0_empirical"]),           fmt(b["p0_empirical"])),
        ("",                        "",                               ""),
        ("ρ  (per-example, direct)",fmt(a["rho_per_example"]),        fmt(b["rho_per_example"])),
        ("ρ  (batch-level)",        fmt(a["rho_batch_level"]),        fmt(b["rho_batch_level"])),
        ("",                        "",                               ""),
        ("P(A=0)  empirical",       fmt(a["p_degenerate_empirical"]), fmt(b["p_degenerate_empirical"])),
        ("P(A=0)  theory (ρ=0.5)", fmt(a["p_degenerate_theory_2pi"]),fmt(b["p_degenerate_theory_2pi"])),
        ("P(A=0)  theory (ρ_emp)", fmt(a["p_degenerate_theory_rho"]),fmt(b["p_degenerate_theory_rho"])),
        ("",                        "",                               ""),
        ("N_min  (δ=0.10)",         nmin_str(a, 0.10),               nmin_str(b, 0.10)),
        ("N_min  (δ=0.05)",         nmin_str(a, 0.05),               nmin_str(b, 0.05)),
        ("N_min  (δ=0.01)",         nmin_str(a, 0.01),               nmin_str(b, 0.01)),
    ]

    for label, va, vb in rows:
        if label == "":
            print()
        else:
            print(f"  {label:>{col}}  {va:>{val}}  {vb:>{val}}")

    print(sep)

    # --- Verdict ---
    rho_a = a["rho_per_example"]
    rho_b = b["rho_per_example"]
    p0_a  = a["p0_empirical"]
    p0_b  = b["p0_empirical"]
    pd_a  = a["p_degenerate_empirical"]
    pd_b  = b["p_degenerate_empirical"]

    print("\n  VERDICT")
    print("-" * (col + val * 2 + 4))

    # Theory predicts malignant regime → high ρ → P(A=0) high
    # "Favourable" = higher ρ (closer to 1)
    if math.isnan(rho_a) and math.isnan(rho_b):
        winner = "  Cannot determine — both ρ values are NaN (low variance in rewards)."
        pick   = None
    elif math.isnan(rho_a):
        winner, pick = lb, b
    elif math.isnan(rho_b):
        winner, pick = la, a
    elif rho_a > rho_b:
        winner, pick = la, a
    elif rho_b > rho_a:
        winner, pick = lb, b
    else:
        winner, pick = "  Tie", None

    if pick is not None:
        delta_rho = abs(rho_a - rho_b)
        print(f"  Higher ρ (incoherent failure mode):  {winner}")
        print(f"  ρ difference:                        {delta_rho:.3f}")
        rho_winner = max(rho_a, rho_b)
        if rho_winner > 0.7:
            verdict = "STRONG — consistent with malignant/incoherent regime (ρ > 0.7)"
        elif rho_winner > 0.5:
            verdict = "MODERATE — suggestive of incoherent regime (ρ > 0.5)"
        elif rho_winner > 0.3:
            verdict = "WEAK — barely distinguishable from benign regime"
        else:
            verdict = "INCONSISTENT — ρ < 0.3, theory prediction not supported"
        print(f"  Theory support:                      {verdict}")
        print()
        print(f"  Recommended setup for paper:")
        print(f"    → Use '{winner}' as the incoherent/malignant regime example.")
        pmodel = pick["model"]
        ptask  = pick["task"]
        print(f"    → Command:")
        print(f"       uv run python -m src.scripts.probe_degeneracy \\")
        print(f"           --task {ptask} --model {pmodel} --K 200 --batch-size 16")
    else:
        print(f"  {winner}")

    # Cross-check p0 interpretation
    print()
    print(f"  p₀ interpretation:")
    for lbl, p0, task in [(la, p0_a, a["task"]), (lb, p0_b, b["task"])]:
        if p0 < 0.1:
            interp = "very low — uninstructed-base regime (N_min could be ~29)"
        elif p0 < 0.25:
            interp = "low — borderline; malignant likely if errors are idiosyncratic"
        elif p0 < 0.5:
            interp = "moderate — capable model; benign if format-correctable"
        else:
            interp = "high — model largely succeeds; format errors may dominate"
        print(f"    {lbl}: p₀={p0:.3f}  →  {interp}")

    print()
    print("  NOTE: ρ > ~0.7 on the winner, combined with p₀ < 0.5, would")
    print("  confirm the incoherent failure mode prediction. Run the benign")
    print("  probe (e.g. Qwen2.5-0.5B / SST-2) to complete the comparison.")
    print(sep)


def main():
    p = argparse.ArgumentParser(description="Compare two degeneracy probe results")
    p.add_argument("--file-a",  required=True, help="Path to first probe JSON")
    p.add_argument("--label-a", default="Setup A", help="Label for first setup")
    p.add_argument("--file-b",  required=True, help="Path to second probe JSON")
    p.add_argument("--label-b", default="Setup B", help="Label for second setup")
    args = p.parse_args()

    a = load(args.file_a)
    b = load(args.file_b)
    print_comparison(a, args.label_a, b, args.label_b)


if __name__ == "__main__":
    main()
