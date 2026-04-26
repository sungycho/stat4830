"""Temperature sampling sweep: majority vote + pass@k + breakdown analysis.

Single inference run per temperature. Saves raw predictions and computes all
metrics and plots in one pass — no need to re-run inference for re-plotting.

Supports:
  boolq    : breakdown = voted_yes / voted_no / abstain + acc by true label
  countdown: breakdown = correct / valid_wrong / format_error

Usage:
  uv run python -m src.scripts.run_temperature_sweep --model facebook/opt-350m --task boolq --val-size 500 --out results/temp_sweep/boolq_350m
  uv run python -m src.scripts.run_temperature_sweep --model Qwen/Qwen2.5-3B-Instruct --task countdown --val-size 200 --max-new-tokens 64 --temperatures 0.7 1.0 1.5 --k-values 1 5 10 20 50 --batch-size 8 --out results/temp_sweep/countdown_qwen3b
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
import numpy as np

from src.backends.hf_backend import HFBackend
from src.tasks import get_task, available_tasks
from src.tasks.countdown import _extract_expression, _safe_eval, _numbers_valid, _extract_numbers_used
from src.utils.seeds import set_seeds

# ── BoolQ helpers ─────────────────────────────────────────────────────────────
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b",  re.IGNORECASE)


def _predict_boolq(text: str) -> str | None:
    yes = _YES_RE.search(text)
    no  = _NO_RE.search(text)
    if yes and no:
        return "yes" if yes.start() < no.start() else "no"
    elif yes:
        return "yes"
    elif no:
        return "no"
    return None


# ── Countdown helpers ─────────────────────────────────────────────────────────

def _predict_countdown(text: str, example: dict) -> float | None:
    expr = _extract_expression(text)
    if expr is None:
        return None
    used = _extract_numbers_used(expr)
    available = [int(n) for n in example["nums"]]
    if not _numbers_valid(used, available):
        return None
    return _safe_eval(expr)


def _countdown_category(text: str, example: dict) -> str:
    """Return 'correct', 'valid_wrong', or 'format_error'."""
    result = _predict_countdown(text, example)
    if result is None:
        return "format_error"
    return "correct" if abs(result - int(example["target"])) < 1e-6 else "valid_wrong"


# ── Task-agnostic prediction ──────────────────────────────────────────────────

def extract_prediction(text: str, example: dict, task_name: str):
    if task_name == "boolq":
        return _predict_boolq(text)
    elif task_name == "countdown":
        return _predict_countdown(text, example)
    raise ValueError(f"Unsupported task: {task_name}")


def is_correct(pred, example: dict, task_name: str) -> bool:
    if pred is None:
        return False
    if task_name == "boolq":
        return pred == {0: "no", 1: "yes"}[example["label"]]
    elif task_name == "countdown":
        return abs(pred - int(example["target"])) < 1e-6
    return False


def majority_vote_pred(preds: list, task_name: str):
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    if task_name == "boolq":
        return Counter(valid).most_common(1)[0][0]
    elif task_name == "countdown":
        rounded = [round(p) for p in valid]
        return float(Counter(rounded).most_common(1)[0][0])
    return None


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_sweep(backend, val_data, task, task_name, k_max, batch_size):
    """Generate k_max samples per example. Returns (all_preds, all_raw_texts)."""
    n = len(val_data)
    all_preds: list[list] = [[] for _ in range(n)]
    all_texts: list[list[str]] = [[] for _ in range(n)]
    prompts = [task.build_prompt(ex) for ex in val_data]

    for k in range(k_max):
        for i in range(0, n, batch_size):
            chunk_prompts  = prompts[i : i + batch_size]
            chunk_examples = val_data[i : i + batch_size]
            outputs = backend.generate_batch(chunk_prompts)
            for j, out in enumerate(outputs):
                all_preds[i + j].append(extract_prediction(out, chunk_examples[j], task_name))
                all_texts[i + j].append(out)
        if (k + 1) % 5 == 0 or (k + 1) == k_max:
            print(f"  sample {k+1}/{k_max}", end="\r")

    print()
    return all_preds, all_texts


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_preds, all_texts, val_data, task_name, k):
    n = len(val_data)
    majority_correct = pass_k_correct = format_err = 0

    # task-specific breakdown counters
    # boolq: voted_yes, voted_no
    # countdown: valid_wrong
    voted_yes = voted_no = valid_wrong = 0
    acc_true_yes = acc_true_no = total_true_yes = total_true_no = 0

    for preds, texts, ex in zip(all_preds, all_texts, val_data):
        subset_preds = preds[:k]
        subset_texts = texts[:k]
        vote = majority_vote_pred(subset_preds, task_name)

        if vote is None:
            format_err += 1
        elif is_correct(vote, ex, task_name):
            majority_correct += 1

        if any(is_correct(p, ex, task_name) for p in subset_preds):
            pass_k_correct += 1

        if task_name == "boolq":
            if vote == "yes":
                voted_yes += 1
            elif vote == "no":
                voted_no += 1
            true = {0: "no", 1: "yes"}[ex["label"]]
            if true == "yes":
                total_true_yes += 1
                if vote == "yes":
                    acc_true_yes += 1
            else:
                total_true_no += 1
                if vote == "no":
                    acc_true_no += 1

        elif task_name == "countdown":
            # count valid_wrong among majority-voted non-None
            if vote is not None and not is_correct(vote, ex, task_name):
                valid_wrong += 1

    result = {
        "k": k,
        "majority_accuracy": majority_correct / n,
        "pass_at_k":         pass_k_correct / n,
        "format_error_rate": format_err / n,
        "majority_correct":  majority_correct,
        "pass_k_correct":    pass_k_correct,
        "format_error":      format_err,
    }

    if task_name == "boolq":
        result["voted_yes_frac"]  = voted_yes / n
        result["voted_no_frac"]   = voted_no / n
        result["voted_none_frac"] = format_err / n
        result["acc_true_yes"]    = acc_true_yes / total_true_yes if total_true_yes else 0
        result["acc_true_no"]     = acc_true_no  / total_true_no  if total_true_no  else 0

    elif task_name == "countdown":
        result["valid_wrong_rate"] = valid_wrong / n
        result["correct_rate"]     = majority_correct / n

    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(results, temperatures, k_values, metric, out_path, dpi, baseline, model, task_name):
    acc_matrix = np.zeros((len(temperatures), len(k_values)))
    for r in results:
        ti = temperatures.index(r["temperature"])
        ki = k_values.index(r["k"])
        acc_matrix[ti, ki] = r[metric]

    vmin = max(0, baseline - 0.05)
    vmax = min(1.0, baseline + 0.35)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(acc_matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=metric.replace("_", " "))
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(range(len(temperatures)))
    ax.set_yticklabels([str(t) for t in temperatures])
    ax.set_xlabel("K (samples per example)", fontsize=10)
    ax.set_ylabel("Temperature", fontsize=10)
    ax.set_title(f"{metric.replace('_',' ').title()} — {model} / {task_name}\n(baseline={baseline:.3f})", fontsize=10)
    for i in range(len(temperatures)):
        for j in range(len(k_values)):
            ax.text(j, i, f"{acc_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_lines(results, temperatures, k_values, out_path, dpi, baseline, model, task_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.cool(np.linspace(0, 1, len(temperatures)))

    for ax, metric, title in zip(axes, ["majority_accuracy", "pass_at_k"], ["Majority Vote Accuracy", "Pass@K"]):
        for temp, color in zip(temperatures, colors):
            vals = [r[metric] for r in results if r["temperature"] == temp]
            ax.plot(k_values, vals, marker="o", label=f"T={temp}", color=color)
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")
        ax.set_xlabel("K (samples per example)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xscale("log")
        ax.set_title(f"{title} — {task_name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(model, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_boolq_bias(results, temperatures, k_values, out_dir, dpi, true_yes_frac):
    """One plot per temperature showing yes/no bias + acc by true label."""
    for temp in temperatures:
        temp_results = [r for r in results if r["temperature"] == temp]
        ks        = [r["k"]               for r in temp_results]
        yes_frac  = [r["voted_yes_frac"]  for r in temp_results]
        no_frac   = [r["voted_no_frac"]   for r in temp_results]
        none_frac = [r["voted_none_frac"] for r in temp_results]
        acc_yes   = [r["acc_true_yes"]    for r in temp_results]
        acc_no    = [r["acc_true_no"]     for r in temp_results]
        overall   = [r["majority_accuracy"] for r in temp_results]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.stackplot(ks, yes_frac, no_frac, none_frac,
                     labels=["Voted Yes", "Voted No", "Abstain"],
                     colors=["#81C784", "#E57373", "#BDBDBD"], alpha=0.5)
        ax.plot(ks, overall, color="black",   linewidth=2,   label="Overall acc",     zorder=5)
        ax.plot(ks, acc_yes, color="#2E7D32", linewidth=1.5, label="Acc on true-yes", zorder=5, linestyle="--")
        ax.plot(ks, acc_no,  color="#C62828", linewidth=1.5, label="Acc on true-no",  zorder=5, linestyle="--")
        ax.axhline(true_yes_frac, color="grey", linewidth=0.8, linestyle=":", label=f"True yes ratio ({true_yes_frac:.2f})")
        ax.set_title(f"Output bias & accuracy breakdown — T={temp} (BoolQ)", fontsize=10)
        ax.set_xlabel("K (samples per example)", fontsize=9)
        ax.set_ylabel("Fraction", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out_path = Path(out_dir) / f"bias_T{str(temp).replace('.','')}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] saved → {out_path}")
        plt.close(fig)


def plot_countdown_breakdown(results, temperatures, k_values, out_dir, dpi):
    """One plot per temperature: correct / valid_wrong / format_error breakdown."""
    for temp in temperatures:
        temp_results = [r for r in results if r["temperature"] == temp]
        ks         = [r["k"]                for r in temp_results]
        correct    = [r["correct_rate"]     for r in temp_results]
        valid_wrong= [r["valid_wrong_rate"] for r in temp_results]
        fmt_err    = [r["format_error_rate"] for r in temp_results]
        pass_k     = [r["pass_at_k"]        for r in temp_results]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.stackplot(ks, correct, valid_wrong, fmt_err,
                     labels=["Correct", "Valid but Wrong", "Format Error"],
                     colors=["#81C784", "#FFB74D", "#E57373"], alpha=0.6)
        ax.plot(ks, pass_k, color="black", linewidth=2, label="Pass@K", zorder=5, linestyle="--")
        ax.set_title(f"Countdown breakdown — T={temp} (Qwen2.5-3B)", fontsize=10)
        ax.set_xlabel("K (samples per example)", fontsize=9)
        ax.set_ylabel("Fraction of val set", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out_path = Path(out_dir) / f"breakdown_T{str(temp).replace('.','')}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] saved → {out_path}")
        plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Temperature sweep with full breakdown analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          default=os.environ.get("MODEL_NAME", "facebook/opt-350m"))
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",          default="bfloat16")
    p.add_argument("--task",           default="boolq", choices=available_tasks())
    p.add_argument("--val-size",       type=int, default=500)
    p.add_argument("--batch-size",     type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8, help="Use 64+ for countdown")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--temperatures",   type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0, 1.5])
    p.add_argument("--k-values",       type=int, nargs="+", default=[1, 5, 10, 20, 50])
    p.add_argument("--out",            default="results/temp_sweep")
    p.add_argument("--dpi",            type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(args.task)
    _, val_data = task.load_data(train_size=1, val_size=args.val_size, seed=args.seed)
    n = len(val_data)

    if args.task == "boolq":
        n_yes = sum(1 for ex in val_data if ex["label"] == 1)
        baseline = max(n_yes, n - n_yes) / n
        true_yes_frac = n_yes / n
        print(f"Val: n={n}  yes={n_yes} ({100*n_yes/n:.1f}%)  majority_baseline={baseline:.3f}")
    else:
        baseline = 0.0
        true_yes_frac = None
        print(f"Val: n={n}  task={args.task}")

    k_max = max(args.k_values)
    all_results = []
    all_raw = {}  # temp -> {"preds": ..., "texts": ...}

    for temp in args.temperatures:
        print(f"\n── Temperature={temp} ──")
        backend = HFBackend(
            model_name=args.model, device=args.device, dtype=args.dtype,
            max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=temp, top_p=0.95,
        )

        t0 = time.perf_counter()
        all_preds, all_texts = run_sweep(backend, val_data, task, args.task, k_max, args.batch_size)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

        # save raw predictions for this temperature
        all_raw[str(temp)] = {
            "preds": [[str(p) if p is not None else None for p in ps] for ps in all_preds],
            "texts": all_texts,
        }

        for k in args.k_values:
            metrics = compute_metrics(all_preds, all_texts, val_data, args.task, k)
            metrics["temperature"] = temp
            all_results.append(metrics)
            if args.task == "boolq":
                print(f"  K={k:4d}  majority={metrics['majority_accuracy']:.3f}  "
                      f"pass@k={metrics['pass_at_k']:.3f}  "
                      f"voted_yes={metrics['voted_yes_frac']:.2f}  "
                      f"acc_no={metrics['acc_true_no']:.3f}")
            else:
                print(f"  K={k:4d}  majority={metrics['majority_accuracy']:.3f}  "
                      f"pass@k={metrics['pass_at_k']:.3f}  "
                      f"valid_wrong={metrics['valid_wrong_rate']:.3f}  "
                      f"format_err={metrics['format_error_rate']:.3f}")

        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # update baseline for countdown using K=1, lowest temp
    if args.task == "countdown":
        k1 = [r for r in all_results if r["k"] == 1 and r["temperature"] == min(args.temperatures)]
        baseline = k1[0]["majority_accuracy"] if k1 else 0.0

    # save everything
    payload = {
        "model": args.model, "task": args.task, "val_size": n,
        "baseline": baseline, "temperatures": args.temperatures,
        "k_values": args.k_values, "results": all_results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "raw_preds.json").write_text(json.dumps(all_raw, indent=2))
    print(f"\n[save] → {out_dir}/results.json + raw_preds.json")

    # plots
    plot_heatmap(all_results, args.temperatures, args.k_values,
                 "majority_accuracy", out_dir / "heatmap_majority.png",
                 args.dpi, baseline, args.model, args.task)
    plot_heatmap(all_results, args.temperatures, args.k_values,
                 "pass_at_k", out_dir / "heatmap_passk.png",
                 args.dpi, baseline, args.model, args.task)
    plot_lines(all_results, args.temperatures, args.k_values,
               out_dir / "lines.png", args.dpi, baseline, args.model, args.task)

    if args.task == "boolq":
        plot_boolq_bias(all_results, args.temperatures, args.k_values,
                        out_dir, args.dpi, true_yes_frac)
    elif args.task == "countdown":
        plot_countdown_breakdown(all_results, args.temperatures, args.k_values,
                                 out_dir, args.dpi)


if __name__ == "__main__":
    main()
