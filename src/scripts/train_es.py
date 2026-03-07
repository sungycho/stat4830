"""GPU-ready ES fine-tuning script. Drop-in replacement for sanity_es_loop.py.

Key additions over sanity_es_loop.py:
- --device: target device (default: cuda if available, else cpu)
- --dtype:  model dtype (default: auto → bfloat16 on GPU, float32 on CPU)
- --perturb-verbose: toggle per-layer prints (default off — kills GPU throughput)

Single-GPU usage:
  uv run python -m src.scripts.train_es --task sst2 --device cuda
  uv run python -m src.scripts.train_es --task rte  --dtype bfloat16

Multi-GPU (TODO):
  Current architecture is single-process single-GPU. To scale to N GPUs:
  - Spawn N worker processes, each holding a model replica on its GPU.
  - Coordinator (GPU 0 or CPU) samples seeds, collects advantages,
    calls es_grad_update on master weights, then broadcasts fresh
    state_dict to workers via torch.distributed or Ray.
  - Each worker evaluates its assigned seeds (+eps/-eps) and returns
    scalar advantages to the coordinator.
  See: torch.distributed, ray.util.collective, or DeepSpeed ZeRO for
  weight broadcast primitives.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

import src.utils.perturb as perturb_module
from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks
from src.utils.seeds import set_seeds
from src.utils.perturb import perturb_inplace, restore_inplace, es_grad_update


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg == "auto":
        return "bfloat16" if device.startswith("cuda") else "float32"
    return dtype_arg


def parse_args():
    p = argparse.ArgumentParser(
        description="ES fine-tuning loop (GPU-ready)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",             default="sst2", choices=available_tasks())
    p.add_argument("--model",            default=os.environ.get("MODEL_NAME", "facebook/opt-1.3b"))
    p.add_argument("--device",           default=_default_device(),
                   help="Target device: cpu | cuda | cuda:0 | cuda:1 ...")
    p.add_argument("--dtype",            default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                   help="Model dtype. 'auto' picks bfloat16 on GPU, float32 on CPU.")
    p.add_argument("--population-size",  type=int,   default=8)
    p.add_argument("--num-iters",        type=int,   default=4)
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--val-every",        type=int,   default=2)
    p.add_argument("--sigma",            type=float, default=1e-3)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--train-size",       type=int,   default=64)
    p.add_argument("--val-size",         type=int,   default=200)
    p.add_argument("--early-stop-delta", type=float, default=0.1,
                   help="Stop if val_acc drops more than this below best (0 = disabled)")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--out-dir",          type=str,   default=None,
                   help="Output dir (default: results/<task>_<timestamp>)")
    p.add_argument("--resume-from",      type=str,   default=None,
                   help="Path to a checkpoint (.pt) to load weights from before training")
    p.add_argument("--perturb-verbose",  action="store_true", default=False,
                   help="Print per-layer progress during perturbation (default off on GPU)")
    # ES variant flags
    p.add_argument("--noise-type",       default="gaussian", choices=["gaussian", "rademacher"],
                   help="Perturbation noise distribution")
    p.add_argument("--one-sided",        action="store_true", default=False,
                   help="One-sided ES: only eval +eps (no antithetic -eps pair)")
    p.add_argument("--no-normalize",     action="store_true", default=False,
                   help="Disable z-score reward normalisation in gradient update")
    p.add_argument("--top-k",            type=int,   default=0,
                   help="ARS-style: keep only top-k seeds by |advantage| before update (0=all)")
    return p.parse_args()


def make_run_dir(args) -> Path:
    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results") / f"{args.task}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_entry(log_path: Path, entry: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_checkpoint(model, path: Path, run_state: dict | None = None) -> None:
    torch.save(model.state_dict(), path)
    if run_state is not None:
        state_path = path.with_suffix(".state.json")
        with open(state_path, "w") as f:
            json.dump(run_state, f)


def load_run_state(checkpoint_path: str) -> dict:
    """Load saved run state (iteration, best_val_acc) alongside a checkpoint.

    Returns empty dict if no state file exists (weights-only warm start).
    Note: RNG state is not saved, so exact episode replay is not guaranteed.
    """
    state_path = Path(checkpoint_path).with_suffix(".state.json")
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def eval_batch(backend, examples: list[dict], task) -> float:
    """Evaluate a mini-batch in a single batched forward pass."""
    prompts = [task.build_prompt(ex) for ex in examples]
    outputs = backend.generate_batch(prompts)
    rewards = [task.score(text, ex) for text, ex in zip(outputs, examples)]
    return sum(rewards) / len(rewards)


def validate(backend, val_data: list[dict], task, batch_size: int = 16) -> float:
    """Validate over the full val set, chunked into batched forward passes."""
    correct = 0
    for i in range(0, len(val_data), batch_size):
        chunk = val_data[i:i + batch_size]
        prompts = [task.build_prompt(ex) for ex in chunk]
        outputs = backend.generate_batch(prompts)
        correct += sum(task.score(text, ex) > 0 for text, ex in zip(outputs, chunk))
    return correct / len(val_data)


def run_es_iteration(
    model, backend, task, train_data, args
) -> tuple[list[int], list[float], int]:
    """Run one ES iteration. Returns (seeds, advantages, fwd_passes_used)."""
    effective_batch = min(args.batch_size, len(train_data))
    if effective_batch < args.batch_size:
        print(f"[warn] batch_size={args.batch_size} > train_size={len(train_data)}, clamped to {effective_batch}")
    batch = random.sample(train_data, effective_batch)
    seeds, advantages = [], []
    fwd_passes = 0

    nt = args.noise_type
    for _ in range(args.population_size):
        seed = random.randint(0, 2**31)

        perturb_inplace(model, seed, args.sigma, sign=+1, noise_type=nt)  # θ → θ+σε
        r_plus = eval_batch(backend, batch, task)
        fwd_passes += effective_batch

        if args.one_sided:
            restore_inplace(model, seed, args.sigma, sign=+1, noise_type=nt)  # θ+σε → θ
            adv = r_plus
        else:
            perturb_inplace(model, seed, 2 * args.sigma, sign=-1, noise_type=nt)  # θ+σε → θ-σε
            r_minus = eval_batch(backend, batch, task)
            fwd_passes += effective_batch
            restore_inplace(model, seed, args.sigma, sign=-1, noise_type=nt)    # θ-σε → θ
            adv = r_plus - r_minus

        seeds.append(seed)
        advantages.append(adv)

    return seeds, advantages, fwd_passes


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # Apply verbose setting globally before any perturb calls
    perturb_module.VERBOSE = args.perturb_verbose

    # Preflight: fail fast if CUDA requested but unavailable
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"[error] --device {args.device} requested but CUDA is not available on this machine.")

    dtype = _resolve_dtype(args.dtype, args.device)
    normalize = not args.no_normalize

    # Preflight: top-k=1 with normalization → z-score undefined after filtering
    if args.top_k == 1 and normalize:
        raise SystemExit(
            "[error] --top-k 1 with normalization requires N≥2 after filtering. "
            "Use --top-k 2 or add --no-normalize."
        )

    run_dir = make_run_dir(args)
    log_path = run_dir / "log.jsonl"
    log_entry(log_path, {"event": "config", **vars(args), "resolved_dtype": dtype, "normalize": normalize})
    print(f"Run dir: {run_dir}")

    t_start = time.perf_counter()
    backend = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        max_new_tokens=4,
        do_sample=False,
    )

    task = get_task(args.task)
    train_data, val_data = task.load_data(
        train_size=args.train_size, val_size=args.val_size, seed=args.seed
    )

    run_state = {}
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=True)
        backend.model.load_state_dict(ckpt)
        run_state = load_run_state(args.resume_from)
        if run_state:
            print(f"Resumed from: {args.resume_from} | iter={run_state.get('iteration', 0)} best_val={run_state.get('best_val_acc', 'n/a')}")
        else:
            print(f"Warm-started weights from: {args.resume_from} (no run state found — iteration/best reset)")

    print(
        f"Task: {args.task}  |  Model: {args.model}\n"
        f"Device: {args.device}  |  Dtype: {dtype}\n"
        f"pop={args.population_size}  iters={args.num_iters}  batch={args.batch_size}  "
        f"sigma={args.sigma}  lr={args.lr}\n"
        f"noise={args.noise_type}  one_sided={args.one_sided}  "
        f"normalize={normalize}  top_k={args.top_k}\n"
        f"train={len(train_data)}  val={len(val_data)}"
    )

    baseline_acc = validate(backend, val_data, task, batch_size=args.batch_size)  # chunked batched eval
    print(f"Baseline val_acc: {baseline_acc:.3f} ({int(baseline_acc * len(val_data))}/{len(val_data)})")
    log_entry(log_path, {"event": "baseline", "val_acc": baseline_acc})

    best_val_acc = run_state.get("best_val_acc", baseline_acc)
    start_iter = run_state.get("iteration", 0)
    # train_fwd: forward passes spent on training perturbations only (the controlled budget).
    # total_fwd: train_fwd + validation passes (informational; not used as x-axis).
    cumulative_train_fwd = run_state.get("cumulative_train_fwd", 0)
    cumulative_total_fwd = run_state.get("cumulative_total_fwd", 0)
    save_checkpoint(backend.model, run_dir / "best.pt", {
        "iteration": start_iter, "best_val_acc": best_val_acc,
        "cumulative_train_fwd": cumulative_train_fwd,
        "cumulative_total_fwd": cumulative_total_fwd,
    })

    for iteration in range(args.num_iters):
        t_iter = time.perf_counter()

        seeds, advantages, iter_fwd = run_es_iteration(backend.model, backend, task, train_data, args)
        es_grad_update(
            backend.model, seeds, advantages,
            lr=args.lr,
            top_k=args.top_k,
            normalize=normalize,
            noise_type=args.noise_type,
        )

        cumulative_train_fwd += iter_fwd
        cumulative_total_fwd += iter_fwd
        iter_time = time.perf_counter() - t_iter
        mean_adv = sum(advantages) / len(advantages)

        entry = {
            "event": "iter",
            "iteration": iteration + 1,
            "mean_adv": mean_adv,
            "advantages": advantages,
            "iter_time": iter_time,
            "iter_fwd": iter_fwd,
            "train_fwd": cumulative_train_fwd,   # x-axis for plots (budget-controlled)
            "total_fwd": cumulative_total_fwd,   # includes val overhead
        }

        print(
            f"iter {iteration + 1}/{args.num_iters} | "
            f"mean_adv={mean_adv:+.4f} | "
            f"advantages={[f'{a:+.3f}' for a in advantages]} | "
            f"train_fwd={cumulative_train_fwd} | iter_time={iter_time:.1f}s"
        )

        current_state = {
            "iteration": start_iter + iteration + 1, "best_val_acc": best_val_acc,
            "cumulative_train_fwd": cumulative_train_fwd,
            "cumulative_total_fwd": cumulative_total_fwd,
        }
        save_checkpoint(backend.model, run_dir / "latest.pt", current_state)

        if (iteration + 1) % args.val_every == 0:
            val_acc = validate(backend, val_data, task, batch_size=args.batch_size)
            cumulative_total_fwd += len(val_data)  # val cost charged to total only
            entry["val_acc"] = val_acc
            entry["total_fwd"] = cumulative_total_fwd  # update total after val

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(backend.model, run_dir / "best.pt", {
                    "iteration": start_iter + iteration + 1, "best_val_acc": best_val_acc,
                    "cumulative_train_fwd": cumulative_train_fwd,
                    "cumulative_total_fwd": cumulative_total_fwd,
                })
                print(f"  >> val_acc={val_acc:.3f} — new best | train_fwd={cumulative_train_fwd}")
            else:
                print(f"  >> val_acc={val_acc:.3f} (best={best_val_acc:.3f}) | train_fwd={cumulative_train_fwd}")

            if args.early_stop_delta > 0 and val_acc < best_val_acc - args.early_stop_delta:
                log_entry(log_path, entry)
                print(
                    f"Early stop: val_acc={val_acc:.3f} dropped >{args.early_stop_delta:.2f} "
                    f"below best={best_val_acc:.3f}"
                )
                break

        log_entry(log_path, entry)

    print(f"\nDone. Total: {time.perf_counter() - t_start:.1f}s | logs → {log_path}")


if __name__ == "__main__":
    main()
