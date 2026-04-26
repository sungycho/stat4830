"""Degeneracy probe: empirically estimate P(A=0) for a given model and task.

Measures the fraction of perturbation pairs that yield zero advantage under
binary accuracy reward. From this, computes the empirical N_min.

Theory: P(A=0) = fraction of seed pairs where r(theta+sigma*eps, B) == r(theta-sigma*eps, B).
        N_min  = ceil(log(alpha_conf) / log(P(A=0)))

Efficiency notes:
- Uses 3-perturb trick: theta -> theta+se -> theta-se -> theta (saves 1 perturb pass vs 4).
- Per-example scores are stored to compute rho directly (not back-calculated).
- Layer-wise in-place perturbation via perturb_inplace (peak memory = largest layer).
- No checkpoint I/O during probe loop.
- Parallel workers: each loads its own model copy; K pairs split across workers.
  On an 80GB GPU, Qwen-0.5B (~1GB) easily fits 4+ copies -> ~4x speedup.

Usage:
  uv run python -m src.scripts.probe_degeneracy --task boolq --model Qwen/Qwen2.5-1.5B-Instruct
  uv run python -m src.scripts.probe_degeneracy --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --K 200 --batch-size 16
  uv run python -m src.scripts.probe_degeneracy --task sst2  --model Qwen/Qwen2.5-1.5B-Instruct --sigma 0.001 --batch-size 16
  uv run python -m src.scripts.probe_degeneracy --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp

from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks, PROMPT_STYLES, resolve_prompt_config
from src.utils.perturb import perturb_inplace, restore_inplace
from src.utils.seeds import set_seeds


# ---------------------------------------------------------------------------
# Dtype resolution (matches train_es.py)
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg == "auto":
        return "bfloat16" if device.startswith("cuda") else "float32"
    return dtype_arg


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

def eval_batch_scores(
    backend, prompts: list[str], examples: list[dict], task, raw: bool = False
) -> tuple[float, list[float]]:
    """Evaluate binary accuracy reward over a batch.

    Returns:
        mean_reward: mean score over the batch (float in [-1, 1] or [0, 1])
        per_example:  list of per-example task.score values (one per example)
    """
    outputs = backend.generate_batch(prompts, raw=raw)
    scores = [task.score(out, ex) for out, ex in zip(outputs, examples)]
    return sum(scores) / len(scores), scores


# ---------------------------------------------------------------------------
# Work item generation (deterministic, done once in main process)
# ---------------------------------------------------------------------------

def _make_work_items(K: int, batch_size: int, probe_pool_size: int, seed: int) -> list:
    """Pre-generate all K (k_idx, seed_val, batch_indices) tuples deterministically."""
    rng = random.Random(seed)
    pool_indices = list(range(probe_pool_size))
    items = []
    for k in range(K):
        seed_val = rng.randint(0, 2**31 - 1)
        batch_indices = rng.sample(pool_indices, min(batch_size, probe_pool_size))
        items.append((k, seed_val, batch_indices))
    return items


# ---------------------------------------------------------------------------
# Worker function (runs in its own process)
# ---------------------------------------------------------------------------

def _worker(rank: int, args_ns, probe_pool: list, work_items: list, result_queue) -> None:
    """Worker process: loads its own model copy and runs its assigned pairs."""
    set_seeds(args_ns.seed + rank)
    dtype = _resolve_dtype(args_ns.dtype, args_ns.device)

    task = get_task(args_ns.task)
    prompt_cfg = resolve_prompt_config(task, getattr(args_ns, "prompt_style", "simple"))
    raw = getattr(args_ns, "no_chat_template", False) or prompt_cfg.force_raw or task.prefer_base_prompt

    backend = create_backend(
        "hf",
        model_name=args_ns.model,
        device=args_ns.device,
        dtype=dtype,
        max_new_tokens=args_ns.max_new_tokens,
        do_sample=False,
    )
    model = backend.model

    local_advantages = []
    local_r_plus    = []
    local_r_minus   = []
    local_x_plus    = []
    local_x_minus   = []
    local_n_degen   = 0

    t_start = time.perf_counter()

    for i, (k_idx, seed_val, batch_indices) in enumerate(work_items):
        t_k    = time.perf_counter()
        batch   = [probe_pool[j] for j in batch_indices]
        prompts = [prompt_cfg.prompt_fn(ex) for ex in batch]

        # ---- 3-perturb trick ----
        perturb_inplace(model, seed_val, args_ns.sigma, sign=+1)
        r_plus, x_plus = eval_batch_scores(backend, prompts, batch, task, raw=raw)

        perturb_inplace(model, seed_val, 2 * args_ns.sigma, sign=-1)
        r_minus, x_minus = eval_batch_scores(backend, prompts, batch, task, raw=raw)

        restore_inplace(model, seed_val, args_ns.sigma, sign=-1)
        # -------------------------

        advantage = r_plus - r_minus
        local_advantages.append(advantage)
        local_r_plus.append(r_plus)
        local_r_minus.append(r_minus)
        local_x_plus.extend(1.0 if s > 0 else 0.0 for s in x_plus)
        local_x_minus.extend(1.0 if s > 0 else 0.0 for s in x_minus)
        if advantage == 0.0:
            local_n_degen += 1

        iter_time = time.perf_counter() - t_k
        elapsed   = time.perf_counter() - t_start
        p_so_far  = local_n_degen / (i + 1)
        eta       = elapsed / (i + 1) * (len(work_items) - i - 1)
        print(
            f"  [W{rank} {i+1:3d}/{len(work_items)}] k={k_idx:4d}  "
            f"t={iter_time:.1f}s  P(A=0)={p_so_far:.3f}  "
            f"adv={advantage:+.4f}  r+={r_plus:.3f}  r-={r_minus:.3f}  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
            flush=True,
        )

    result_queue.put({
        "rank":        rank,
        "k_indices":   [item[0] for item in work_items],
        "advantages":  local_advantages,
        "r_plus":      local_r_plus,
        "r_minus":     local_r_minus,
        "x_plus_all":  local_x_plus,
        "x_minus_all": local_x_minus,
        "n_degenerate": local_n_degen,
    })


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(args):
    set_seeds(args.seed)

    dtype = _resolve_dtype(args.dtype, args.device)

    # --- load task and data ---
    task = get_task(args.task)
    train_data, val_data = task.load_data(
        train_size=args.probe_size,
        val_size=args.probe_size,
        seed=args.seed,
    )
    probe_pool = train_data
    print(f"[probe] task={args.task}  probe_pool={len(probe_pool)} examples")

    num_workers = args.num_workers
    print(f"\n[probe] sigma={args.sigma}  K={args.K}  B={args.batch_size}  dtype={dtype}  workers={num_workers}")
    print(f"[probe] {args.K} probe pairs  =  {2 * args.K * args.batch_size} forward passes")
    print(f"[probe] efficiency: 3-perturb trick (theta->+se->-se->theta)")
    if num_workers > 1:
        print(f"[probe] parallel: {num_workers} workers, ~{args.K // num_workers} pairs each")
    print()

    # Pre-generate all work items deterministically
    all_work_items = _make_work_items(args.K, args.batch_size, len(probe_pool), args.seed)

    # Split into chunks for each worker
    chunks = [all_work_items[i::num_workers] for i in range(num_workers)]

    t_probe_start = time.perf_counter()

    if num_workers == 1:
        # Single-process path (avoids spawn overhead for small runs)
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        _worker(0, args, probe_pool, chunks[0], result_queue)
        raw_results = [result_queue.get()]
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for rank in range(num_workers):
            p = ctx.Process(
                target=_worker,
                args=(rank, args, probe_pool, chunks[rank], result_queue),
            )
            p.start()
            processes.append(p)

        raw_results = []
        for _ in range(num_workers):
            raw_results.append(result_queue.get())

        for p in processes:
            p.join()

    # --- aggregate results in k_idx order ---
    # Build a mapping k_idx -> result entry
    per_k = {}
    for worker_res in raw_results:
        for i, k_idx in enumerate(worker_res["k_indices"]):
            per_k[k_idx] = {
                "advantage": worker_res["advantages"][i],
                "r_plus":    worker_res["r_plus"][i],
                "r_minus":   worker_res["r_minus"][i],
            }

    advantages   = [per_k[k]["advantage"] for k in range(args.K)]
    r_plus_list  = [per_k[k]["r_plus"]    for k in range(args.K)]
    r_minus_list = [per_k[k]["r_minus"]   for k in range(args.K)]

    # Per-example scores are already binary lists; concatenate in original k order
    # We need to reconstruct per-k x_plus/x_minus from workers.
    # Workers stored them as flat lists in their k order, so re-collect per-k.
    # Rebuild: for each worker, zip k_indices with chunks of B scores.
    B = args.batch_size
    x_plus_all  = []
    x_minus_all = []
    # Build per-k x_plus/x_minus from worker data
    per_k_x = {}
    for worker_res in raw_results:
        # Each worker has flat x_plus_all of length len(work_items)*B
        xp = worker_res["x_plus_all"]
        xm = worker_res["x_minus_all"]
        for i, k_idx in enumerate(worker_res["k_indices"]):
            per_k_x[k_idx] = (xp[i*B:(i+1)*B], xm[i*B:(i+1)*B])

    for k in range(args.K):
        xp, xm = per_k_x[k]
        x_plus_all.extend(xp)
        x_minus_all.extend(xm)

    n_degenerate = sum(1 for a in advantages if a == 0.0)

    # --- compute results ---
    p_degenerate = n_degenerate / args.K

    conf_levels = [0.10, 0.05, 0.01]
    nmin_table  = {}
    for alpha_conf in conf_levels:
        if p_degenerate == 0.0:
            nmin = 1
        elif p_degenerate >= 1.0:
            nmin = float("inf")
        else:
            nmin = math.ceil(math.log(alpha_conf) / math.log(p_degenerate))
        nmin_table[alpha_conf] = nmin

    # --- estimate base model accuracy p0 ---
    print(f"\n[probe] estimating base model accuracy p0 over full probe_pool ({len(probe_pool)} examples)...")
    backend_p0 = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    task_p0 = get_task(args.task)
    prompt_cfg_p0 = resolve_prompt_config(task_p0, getattr(args, "prompt_style", "simple"))
    raw_p0 = getattr(args, "no_chat_template", False) or prompt_cfg_p0.force_raw or task_p0.prefer_base_prompt
    base_prompts = [prompt_cfg_p0.prompt_fn(ex) for ex in probe_pool]
    _, base_scores = eval_batch_scores(backend_p0, base_prompts, probe_pool, task_p0, raw=raw_p0)
    p0 = sum(1.0 for s in base_scores if s > 0) / len(base_scores)

    # --- rho: batch-level ---
    n_k       = len(r_plus_list)
    mu_rp     = sum(r_plus_list)  / n_k
    mu_rm     = sum(r_minus_list) / n_k
    cov_batch = sum((r_plus_list[i] - mu_rp) * (r_minus_list[i] - mu_rm) for i in range(n_k)) / n_k
    var_rp    = sum((x - mu_rp) ** 2 for x in r_plus_list)  / n_k
    var_rm    = sum((x - mu_rm) ** 2 for x in r_minus_list) / n_k
    rho_batch = cov_batch / math.sqrt(var_rp * var_rm) if var_rp > 0 and var_rm > 0 else float("nan")

    # --- rho: per-example (direct) ---
    n_ex    = len(x_plus_all)
    mu_xp   = sum(x_plus_all)  / n_ex
    mu_xm   = sum(x_minus_all) / n_ex
    cov_ex  = sum((x_plus_all[i] - mu_xp) * (x_minus_all[i] - mu_xm) for i in range(n_ex)) / n_ex
    var_xp  = sum((x - mu_xp) ** 2 for x in x_plus_all)  / n_ex
    var_xm  = sum((x - mu_xm) ** 2 for x in x_minus_all) / n_ex
    rho_per_example = cov_ex / math.sqrt(var_xp * var_xm) if var_xp > 0 and var_xm > 0 else float("nan")

    # --- theoretical P(A=0) ---
    denom_2pi    = 2 * math.pi * args.batch_size * p0 * (1 - p0)
    p_theory_2pi = 1.0 / math.sqrt(denom_2pi) if denom_2pi > 0 else float("nan")

    denom_rho    = 4 * math.pi * args.batch_size * p0 * (1 - p0) * (1 - rho_per_example)
    p_theory_rho = 1.0 / math.sqrt(denom_rho) if denom_rho > 0 else float("nan")

    # --- report ---
    total_time = time.perf_counter() - t_probe_start
    print("\n" + "=" * 65)
    print(f"DEGENERACY PROBE RESULTS")
    print("=" * 65)
    print(f"  Model:                  {args.model}")
    print(f"  Task:                   {args.task}")
    print(f"  sigma:                  {args.sigma}")
    print(f"  K (probe pairs):        {args.K}")
    print(f"  B (batch size):         {args.batch_size}")
    print(f"  Workers:                {num_workers}")
    print(f"  Total wall-clock time:  {total_time:.1f}s  ({total_time/args.K:.1f}s/pair)")
    print()
    print(f"  Base accuracy p0:       {p0:.3f}")
    print()
    print(f"  rho (per-example):      {rho_per_example:.3f}  "
          f"[Corr(X_j+, X_j-) pooled over {n_ex} example pairs — direct measurement]")
    print(f"  rho (batch-level):      {rho_batch:.3f}  "
          f"[Corr(r+, r-) over {n_k} batch pairs — equals per-example rho under i.i.d.]")
    print()
    print(f"  P(A=0) empirical:       {p_degenerate:.3f}  ({n_degenerate}/{args.K} degenerate pairs)")
    print(f"  P(A=0) theory (rho=0.5):{p_theory_2pi:.3f}  "
          f"[1/sqrt(2*pi*B*p0*(1-p0))]")
    print(f"  P(A=0) theory (rho_emp):{p_theory_rho:.3f}  "
          f"[1/sqrt(4*pi*B*p0*(1-p0)*(1-rho))]  error={abs(p_theory_rho - p_degenerate)/p_degenerate*100:.1f}%")
    print()
    print(f"  Advantage stats (nonzero):")
    nonzero = [a for a in advantages if a != 0.0]
    if nonzero:
        print(f"    count:      {len(nonzero)}/{args.K}")
        print(f"    mean |A|:   {sum(abs(a) for a in nonzero)/len(nonzero):.4f}")
        print(f"    max  |A|:   {max(abs(a) for a in advantages):.4f}")
    print()
    print(f"  N_min estimates:")
    for alpha_conf, nmin in nmin_table.items():
        marker = "  <-- paper uses" if alpha_conf == 0.05 else ""
        print(f"    delta={alpha_conf:.2f}  ->  N_min = {nmin}{marker}")
    print("=" * 65)

    # --- save results ---
    results = {
        "model":                        args.model,
        "task":                         args.task,
        "sigma":                        args.sigma,
        "K":                            args.K,
        "batch_size":                   args.batch_size,
        "num_workers":                  num_workers,
        "seed":                         args.seed,
        "total_wall_clock_s":           total_time,
        "p0_empirical":                 p0,
        "p_degenerate_empirical":       p_degenerate,
        "p_degenerate_theory_2pi":      p_theory_2pi,
        "p_degenerate_theory_rho":      p_theory_rho,
        "rho_per_example":              rho_per_example,
        "rho_batch_level":              rho_batch,
        "n_degenerate":                 n_degenerate,
        "nmin_table":                   {str(k): v for k, v in nmin_table.items()},
        "advantages":                   advantages,
        "r_plus":                       r_plus_list,
        "r_minus":                      r_minus_list,
        "x_plus_all":                   x_plus_all,
        "x_minus_all":                  x_minus_all,
    }

    out_path = Path(args.output) if args.output else Path(
        f"results/degen_probe/probe_{args.task}_{args.model.replace('/', '_')}"
        f"_sigma{args.sigma}_K{args.K}_B{args.batch_size}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    p.add_argument("--task",         required=True, choices=available_tasks(),
                   help="Task to probe")
    p.add_argument("--model",        default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--sigma",        type=float, default=0.001,
                   help="Perturbation scale (same as training)")
    p.add_argument("--K",            type=int, default=200,
                   help="Number of perturbation pairs to sample")
    p.add_argument("--batch-size",   type=int, default=16,
                   help="Batch size per evaluation (same as training)")
    p.add_argument("--probe-size",   type=int, default=500,
                   help="Pool of examples to sample batches from")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",        default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="Model dtype. 'auto' picks bfloat16 on CUDA, float32 on CPU.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num-workers",  type=int, default=1,
                   help="Number of parallel worker processes. Each loads its own model copy. "
                        "Use 4-8 on an 80GB GPU with Qwen-0.5B.")
    p.add_argument("--output",       default=None,
                   help="Path to save JSON results")
    p.add_argument("--prompt-style", default="simple", choices=PROMPT_STYLES,
                   help="Prompt template style (simple/mezo/free)")
    p.add_argument("--no-chat-template", action="store_true", default=False,
                   help="Skip chat template (use for base/uninstructed models)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(args)
