"""Central ES fine-tuning script. Task, model, and all hyperparameters are CLI-configurable.

Run with:
  uv run python -m src.scripts.sanity_es_loop --help
  uv run python -m src.scripts.sanity_es_loop --task sst2 --num-iters 10 --lr 1e-4
  uv run python -m src.scripts.sanity_es_loop --task rte  --batch-size 8
  uv run python -m src.scripts.sanity_es_loop --task boolq --population-size 4

Design:
- Antithetic pairs (3-pass per pair) for variance reduction.
- Mini-batch sampled once per iteration, reused for all perturbations in that
  iteration — keeps antithetic advantage r+-r- free from batch sampling noise.
- Single aggregated es_grad_update per iteration.
- Checkpoints saved every iteration (latest.pt) and on val improvement (best.pt).
- Early stopping if val_acc drops sharply below best.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks
from src.utils.seeds import set_seeds
from src.utils.perturb import perturb_inplace, restore_inplace, es_grad_update


def parse_args():
    p = argparse.ArgumentParser(
        description="ES fine-tuning loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",            default="sst2", choices=available_tasks(),
                   help="Classification task to train on")
    p.add_argument("--model",           default=os.environ.get("MODEL_NAME", "facebook/opt-1.3b"))
    p.add_argument("--population-size", type=int,   default=8)
    p.add_argument("--num-iters",       type=int,   default=4)
    p.add_argument("--batch-size",      type=int,   default=16)
    p.add_argument("--val-every",       type=int,   default=2)
    p.add_argument("--sigma",           type=float, default=1e-3)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--train-size",      type=int,   default=64)
    p.add_argument("--val-size",        type=int,   default=200)
    p.add_argument("--early-stop-delta",type=float, default=0.1,
                   help="Stop if val_acc drops more than this below best (0 = disabled)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--out-dir",         type=str,   default=None,
                   help="Output dir (default: results/<task>_<timestamp>)")
    p.add_argument("--resume-from",     type=str,   default=None,
                   help="Path to a checkpoint (.pt) to load weights from before training")
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


def save_checkpoint(model, path: Path) -> None:
    torch.save(model.state_dict(), path)


def eval_batch(backend, examples: list[dict], task) -> float:
    rewards = [
        task.score(backend.generate(task.build_prompt(ex)), ex)
        for ex in examples
    ]
    return sum(rewards) / len(rewards)


def validate(backend, val_data: list[dict], task) -> float:
    correct = sum(
        1 for ex in val_data
        if task.score(backend.generate(task.build_prompt(ex)), ex) > 0
    )
    return correct / len(val_data)


def run_es_iteration(model, backend, task, train_data, args) -> tuple[list[int], list[float]]:
    """Sample a fixed mini-batch, run population_size antithetic pairs on it."""
    effective_batch = min(args.batch_size, len(train_data))
    if effective_batch < args.batch_size:
        print(f"[warn] batch_size={args.batch_size} > train_size={len(train_data)}, clamped to {effective_batch}")
    batch = random.sample(train_data, effective_batch)
    seeds, advantages = [], []

    for _ in range(args.population_size):
        seed = random.randint(0, 2**31)

        perturb_inplace(model, seed, args.sigma, sign=+1)        # θ → θ+σε
        r_plus = eval_batch(backend, batch, task)
        perturb_inplace(model, seed, 2 * args.sigma, sign=-1)    # θ+σε → θ-σε
        r_minus = eval_batch(backend, batch, task)
        restore_inplace(model, seed, args.sigma, sign=-1)         # θ-σε → θ

        seeds.append(seed)
        advantages.append(r_plus - r_minus)

    return seeds, advantages


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    run_dir = make_run_dir(args)
    log_path = run_dir / "log.jsonl"
    log_entry(log_path, {"event": "config", **vars(args)})
    print(f"Run dir: {run_dir}")

    t_start = time.perf_counter()
    backend = create_backend(
        "hf",
        model_name=args.model,
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        do_sample=False,
    )

    task = get_task(args.task)
    train_data, val_data = task.load_data(
        train_size=args.train_size, val_size=args.val_size, seed=args.seed
    )

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        backend.model.load_state_dict(ckpt)
        print(f"Resumed weights from: {args.resume_from}")

    print(
        f"Task: {args.task}  |  Model: {args.model}\n"
        f"pop={args.population_size}  iters={args.num_iters}  batch={args.batch_size}  "
        f"sigma={args.sigma}  lr={args.lr}\n"
        f"train={len(train_data)}  val={len(val_data)}"
    )

    baseline_acc = validate(backend, val_data, task)
    print(f"Baseline val_acc: {baseline_acc:.3f} ({int(baseline_acc * len(val_data))}/{len(val_data)})")
    log_entry(log_path, {"event": "baseline", "val_acc": baseline_acc})

    best_val_acc = baseline_acc
    save_checkpoint(backend.model, run_dir / "best.pt")

    for iteration in range(args.num_iters):
        t_iter = time.perf_counter()

        seeds, advantages = run_es_iteration(backend.model, backend, task, train_data, args)
        es_grad_update(backend.model, seeds, advantages, lr=args.lr)

        iter_time = time.perf_counter() - t_iter
        mean_adv = sum(advantages) / len(advantages)

        entry = {
            "event": "iter",
            "iteration": iteration + 1,
            "mean_adv": mean_adv,
            "advantages": advantages,
            "iter_time": iter_time,
        }

        print(
            f"iter {iteration + 1}/{args.num_iters} | "
            f"mean_adv={mean_adv:+.4f} | "
            f"advantages={[f'{a:+.3f}' for a in advantages]} | "
            f"iter_time={iter_time:.1f}s"
        )

        save_checkpoint(backend.model, run_dir / "latest.pt")

        if (iteration + 1) % args.val_every == 0:
            val_acc = validate(backend, val_data, task)
            entry["val_acc"] = val_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(backend.model, run_dir / "best.pt")
                print(f"  >> val_acc={val_acc:.3f} — new best, checkpoint saved")
            else:
                print(f"  >> val_acc={val_acc:.3f} (best={best_val_acc:.3f})")

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
