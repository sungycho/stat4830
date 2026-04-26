"""Analyze output bias in temperature sweep results for BoolQ.

For each (temperature, K), computes:
  - fraction of majority-voted answers that are "yes" vs "no" vs abstain
  - accuracy broken down by true label (yes questions vs no questions)
  - whether accuracy is explained by yes-bias

Reads the raw predictions by re-running inference from saved results.json,
OR re-runs inference if --rerun is set.

Since run_temperature_sweep.py doesn't save raw per-example predictions
(only aggregate metrics), this script re-runs a lightweight version.

Usage:
  uv run python -m src.scripts.analyze_temp_sweep --results-dir results/temp_sweep/boolq_350m --val-size 500 --out results/temp_sweep/boolq_350m/bias_analysis.png
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.backends.hf_backend import HFBackend
from src.tasks import get_task
from src.utils.seeds import set_seeds

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b",  re.IGNORECASE)


def predict(text: str) -> str | None:
    yes = _YES_RE.search(text)
    no  = _NO_RE.search(text)
    if yes and no:
        return "yes" if yes.start() < no.start() else "no"
    elif yes:
        return "yes"
    elif no:
        return "no"
    return None


def majority_vote(preds: list[str | None]) -> str | None:
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def run_sweep(backend, val_data, task, k_max, batch_size):
    n = len(val_data)
    all_preds = [[] for _ in range(n)]
    prompts = [task.build_prompt(ex) for ex in val_data]
    for k in range(k_max):
        for i in range(0, n, batch_size):
            outputs = backend.generate_batch(prompts[i : i + batch_size])
            for j, out in enumerate(outputs):
                all_preds[i + j].append(predict(out))
        if (k + 1) % 5 == 0 or (k + 1) == k_max:
            print(f"  sample {k+1}/{k_max}", end="\r")
    print()
    return all_preds


def analyze(all_preds, val_data, k_values):
    """For each K, compute majority vote bias and accuracy by true label."""
    label_map = {0: "no", 1: "yes"}
    results = []
    for k in k_values:
        voted_yes = voted_no = voted_none = 0
        correct_true_yes = correct_true_no = 0
        total_true_yes = total_true_no = 0

        for preds, ex in zip(all_preds, val_data):
            vote = majority_vote(preds[:k])
            true = label_map[ex["label"]]

            if vote == "yes":
                voted_yes += 1
            elif vote == "no":
                voted_no += 1
            else:
                voted_none += 1

            if true == "yes":
                total_true_yes += 1
                if vote == "yes":
                    correct_true_yes += 1
            else:
                total_true_no += 1
                if vote == "no":
                    correct_true_no += 1

        n = len(val_data)
        results.append({
            "k": k,
            "voted_yes_frac":    voted_yes  / n,
            "voted_no_frac":     voted_no   / n,
            "voted_none_frac":   voted_none / n,
            "acc_true_yes":      correct_true_yes / total_true_yes if total_true_yes else 0,
            "acc_true_no":       correct_true_no  / total_true_no  if total_true_no  else 0,
            "overall_acc":       (correct_true_yes + correct_true_no) / n,
        })
    return results


def plot(all_temp_results, temperatures, k_values, out_path, dpi, val_data):
    n_yes = sum(1 for ex in val_data if ex["label"] == 1)
    n = len(val_data)
    true_yes_frac = n_yes / n

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One figure per temperature
    for ti, (temp, res_list) in enumerate(zip(temperatures, all_temp_results)):
        fig, ax = plt.subplots(figsize=(7, 4))

        ks        = [r["k"]              for r in res_list]
        yes_frac  = [r["voted_yes_frac"] for r in res_list]
        no_frac   = [r["voted_no_frac"]  for r in res_list]
        none_frac = [r["voted_none_frac"] for r in res_list]
        acc_yes   = [r["acc_true_yes"]   for r in res_list]
        acc_no    = [r["acc_true_no"]    for r in res_list]
        overall   = [r["overall_acc"]    for r in res_list]

        ax.stackplot(ks, yes_frac, no_frac, none_frac,
                     labels=["Voted Yes", "Voted No", "Abstain"],
                     colors=["#81C784", "#E57373", "#BDBDBD"],
                     alpha=0.5)

        ax.plot(ks, overall, color="black",   linewidth=2,   label="Overall acc",     zorder=5)
        ax.plot(ks, acc_yes, color="#2E7D32", linewidth=1.5, label="Acc on true-yes", zorder=5, linestyle="--")
        ax.plot(ks, acc_no,  color="#C62828", linewidth=1.5, label="Acc on true-no",  zorder=5, linestyle="--")
        ax.axhline(true_yes_frac, color="grey", linewidth=0.8, linestyle=":",
                   label=f"True yes ratio ({true_yes_frac:.2f})")

        ax.set_title(f"Output bias & accuracy breakdown — Temperature={temp} (BoolQ, OPT-350M)",
                     fontsize=10)
        ax.set_xlabel("K (samples per example)", fontsize=9)
        ax.set_ylabel("Fraction", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        temp_path = out_path.parent / f"bias_T{str(temp).replace('.', '')}.png"
        fig.savefig(temp_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] saved → {temp_path}")
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory containing results.json from run_temperature_sweep.py")
    p.add_argument("--model",       default=None,
                   help="Override model from results.json")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",       default="bfloat16")
    p.add_argument("--val-size",    type=int, default=500)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default=None,
                   help="Output plot path (default: <results-dir>/bias_analysis.png)")
    p.add_argument("--dpi",         type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    results_path = Path(args.results_dir) / "results.json"
    saved = json.loads(results_path.read_text())
    model       = args.model or saved["model"]
    temperatures = saved["temperatures"]
    k_values     = saved["k_values"]
    out_path     = args.out or str(Path(args.results_dir) / "bias_analysis.png")

    task = get_task("boolq")
    _, val_data = task.load_data(train_size=1, val_size=args.val_size, seed=args.seed)

    k_max = max(k_values)
    all_temp_results = []

    for temp in temperatures:
        print(f"\n── Temperature={temp} ──")
        backend = HFBackend(
            model_name=model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.95,
        )
        t0 = time.perf_counter()
        all_preds = run_sweep(backend, val_data, task, k_max, args.batch_size)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

        res = analyze(all_preds, val_data, k_values)
        all_temp_results.append(res)

        for r in res:
            print(f"  K={r['k']:4d}  voted_yes={r['voted_yes_frac']:.2f}  "
                  f"voted_no={r['voted_no_frac']:.2f}  "
                  f"acc_true_yes={r['acc_true_yes']:.3f}  "
                  f"acc_true_no={r['acc_true_no']:.3f}  "
                  f"overall={r['overall_acc']:.3f}")

        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot(all_temp_results, temperatures, k_values, out_path, args.dpi, val_data)


if __name__ == "__main__":
    main()
