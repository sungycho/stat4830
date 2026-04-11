"""Degeneracy probe: empirically estimate P(A=0) for a given model and task.

Measures the fraction of perturbation pairs that yield zero advantage under
binary accuracy reward. From this, computes the empirical N_min.

Theory: P(A=0) = fraction of seed pairs where r(theta+sigma*eps, B) == r(theta-sigma*eps, B).
        N_min  = ceil(log(alpha_conf) / log(P(A=0)))

Usage:
  uv run python -m src.scripts.probe_degeneracy --task boolq --model Qwen/Qwen2.5-1.5B-Instruct
  uv run python -m src.scripts.probe_degeneracy --task trec  --model Qwen/Qwen2.5-0.5B-Instruct --K 300
  uv run python -m src.scripts.probe_degeneracy --task sst2  --model Qwen/Qwen2.5-1.5B-Instruct --sigma 0.001 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks
from src.utils.perturb import perturb_inplace, restore_inplace
from src.utils.seeds import set_seeds


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

def eval_batch(backend, prompts: list[str], examples: list[dict], task) -> float:
    """Evaluate binary accuracy reward over a batch. Returns mean ±1 score."""
    outputs = backend.generate_batch(prompts)
    scores = [task.score(out, ex) for out, ex in zip(outputs, examples)]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(args):
    set_seeds(args.seed)

    # --- load task and data ---
    task = get_task(args.task)
    train_data, val_data = task.load_data(
        train_size=args.probe_size,
        val_size=args.probe_size,
        seed=args.seed,
    )
    # Use train split as the probe pool (same as training would see)
    probe_pool = train_data
    print(f"[probe] task={args.task}  probe_pool={len(probe_pool)} examples")

    # --- load model ---
    backend = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    # --- collect flat parameter list for perturbation ---
    # We perturb all parameters jointly (same as train_es.py)
    model = backend.model

    print(f"\n[probe] sigma={args.sigma}  K={args.K}  B={args.batch_size}")
    print(f"[probe] running {args.K} perturbation pairs ({2 * args.K * args.batch_size} total forward passes)...\n")

    n_degenerate = 0
    advantages = []

    for k in range(args.K):
        # Sample a fresh mini-batch
        batch = random.sample(probe_pool, min(args.batch_size, len(probe_pool)))
        prompts = [task.build_prompt(ex) for ex in batch]

        # Sample perturbation direction
        seed_val = random.randint(0, 2**31 - 1)
        torch.manual_seed(seed_val)

        # Evaluate r+
        perturb_inplace(model, seed_val, args.sigma, sign=+1)
        r_plus = eval_batch(backend, prompts, batch, task)
        restore_inplace(model, seed_val, args.sigma, sign=+1)

        # Evaluate r-
        perturb_inplace(model, seed_val, args.sigma, sign=-1)
        r_minus = eval_batch(backend, prompts, batch, task)
        restore_inplace(model, seed_val, args.sigma, sign=-1)

        advantage = r_plus - r_minus
        advantages.append(advantage)

        if advantage == 0.0:
            n_degenerate += 1

        if (k + 1) % 50 == 0:
            p_so_far = n_degenerate / (k + 1)
            print(f"  [{k+1:4d}/{args.K}]  P(A=0) so far = {p_so_far:.3f}  "
                  f"(degenerate: {n_degenerate}/{k+1})")

    # --- compute results ---
    p_degenerate = n_degenerate / args.K

    # N_min for various confidence levels
    conf_levels = [0.10, 0.05, 0.01]
    nmin_table = {}
    for alpha_conf in conf_levels:
        if p_degenerate == 0.0:
            nmin = 1  # any N works
        elif p_degenerate >= 1.0:
            nmin = float("inf")
        else:
            nmin = math.ceil(math.log(alpha_conf) / math.log(p_degenerate))
        nmin_table[alpha_conf] = nmin

    # Theoretical P(A=0) from formula (requires p0 estimate)
    # Estimate p0 from base model accuracy
    print("\n[probe] estimating base model accuracy p0...")
    base_batch = random.sample(probe_pool, min(args.batch_size * 4, len(probe_pool)))
    base_prompts = [task.build_prompt(ex) for ex in base_batch]
    base_outputs = backend.generate_batch(base_prompts)
    base_scores = [task.score(out, ex) for out, ex in zip(base_outputs, base_batch)]
    # score is +1 or -1; convert to 0/1
    p0 = sum(1 for s in base_scores if s > 0) / len(base_scores)

    # Theoretical P(A=0) from normal approximation
    denom = 2 * math.pi * args.batch_size * p0 * (1 - p0)
    p_theoretical = 1.0 / math.sqrt(denom) if denom > 0 else float("nan")

    # --- report ---
    print("\n" + "=" * 60)
    print(f"DEGENERACY PROBE RESULTS")
    print("=" * 60)
    print(f"  Model:             {args.model}")
    print(f"  Task:              {args.task}")
    print(f"  sigma:             {args.sigma}")
    print(f"  K (probe pairs):   {args.K}")
    print(f"  B (batch size):    {args.batch_size}")
    print()
    print(f"  Base accuracy p0:  {p0:.3f}")
    print(f"  P(A=0) empirical:  {p_degenerate:.3f}")
    print(f"  P(A=0) theory:     {p_theoretical:.3f}  "
          f"[formula: 1/sqrt(2*pi*B*p0*(1-p0))]")
    print()
    print(f"  Advantage stats:")
    nonzero = [a for a in advantages if a != 0.0]
    if nonzero:
        print(f"    mean |A| (nonzero): {sum(abs(a) for a in nonzero)/len(nonzero):.4f}")
        print(f"    max  |A|:           {max(abs(a) for a in advantages):.4f}")
    print()
    print(f"  N_min estimates:")
    for alpha_conf, nmin in nmin_table.items():
        marker = "  <-- paper uses" if alpha_conf == 0.05 else ""
        print(f"    alpha={alpha_conf:.2f}  ->  N_min = {nmin}{marker}")
    print("=" * 60)

    # --- save results ---
    results = {
        "model": args.model,
        "task": args.task,
        "sigma": args.sigma,
        "K": args.K,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "p0_empirical": p0,
        "p_degenerate_empirical": p_degenerate,
        "p_degenerate_theoretical": p_theoretical,
        "n_degenerate": n_degenerate,
        "nmin_table": {str(k): v for k, v in nmin_table.items()},
        "advantages": advantages,
    }

    out_path = Path(args.output) if args.output else Path(
        f"results/probe_{args.task}_{args.model.replace('/', '_')}_sigma{args.sigma}_K{args.K}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Degeneracy probe: estimate P(A=0) empirically")
    p.add_argument("--task",       required=True, choices=available_tasks(),
                   help="Task to probe")
    p.add_argument("--model",      default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--sigma",      type=float, default=0.001,
                   help="Perturbation scale (same as training)")
    p.add_argument("--K",          type=int, default=200,
                   help="Number of perturbation pairs to sample")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size per evaluation (same as training)")
    p.add_argument("--probe-size", type=int, default=500,
                   help="Pool of examples to sample batches from")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",      default="auto")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default=None,
                   help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(args)
