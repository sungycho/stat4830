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
import re
import random
import time
from pathlib import Path

import torch

import src.utils.perturb as perturb_module
from src.backends.factory import create_backend
from src.tasks import get_task, available_tasks
from src.utils.seeds import set_seeds
from src.utils.perturb import perturb_inplace, restore_inplace, es_grad_update

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b",  re.IGNORECASE)


def _classify_output(text: str, example: dict) -> str:
    """Classify a single output as 'correct', 'wrong_answer', or 'format_error'."""
    yes = _YES_RE.search(text)
    no  = _NO_RE.search(text)
    if yes and no:
        pred = 1 if yes.start() < no.start() else 0
    elif yes:
        pred = 1
    elif no:
        pred = 0
    else:
        return "format_error"
    return "correct" if pred == example["label"] else "wrong_answer"


def decomp_eval(backend, val_data: list[dict], task, batch_size: int,
                base_categories: list[str] | None,
                prompt_style: str = "default") -> dict:
    """Run full per-example decomposition eval over val_data.

    Returns aggregate counts and per-example categories.
    base_categories: list of 'correct'/'wrong_answer'/'format_error' from base model run,
                     indexed identically to val_data. If None, skips decomposition counts.
    """
    prompt_fn = _prompt_fn(task, backend, prompt_style)
    raw = task.prefer_base_prompt or prompt_style == "mezo"
    categories = []
    predictions = []  # "yes", "no", "format_error"
    for i in range(0, len(val_data), batch_size):
        chunk   = val_data[i : i + batch_size]
        prompts = [prompt_fn(ex) for ex in chunk]
        outputs = backend.generate_batch(prompts, raw=raw)
        for ex, out in zip(chunk, outputs):
            categories.append(_classify_output(out, ex))
            yes = _YES_RE.search(out)
            no  = _NO_RE.search(out)
            if yes and no:
                predictions.append("yes" if yes.start() < no.start() else "no")
            elif yes:
                predictions.append("yes")
            elif no:
                predictions.append("no")
            else:
                predictions.append("format_error")

    n = len(val_data)
    correct      = categories.count("correct")
    wrong_answer = categories.count("wrong_answer")
    format_error = categories.count("format_error")

    result = {
        "n": n,
        "correct": correct,
        "wrong_answer": wrong_answer,
        "format_error": format_error,
        "val_acc": correct / n,
        "pred_yes": predictions.count("yes"),
        "pred_no": predictions.count("no"),
        "pred_format": predictions.count("format_error"),
        "categories": categories,
    }

    if base_categories is not None:
        strictly_correct = reasoning_thicket = format_thicket = regression = 0
        for base_cat, method_cat in zip(base_categories, categories):
            base_ok   = base_cat   == "correct"
            method_ok = method_cat == "correct"
            if base_ok and method_ok:
                strictly_correct += 1
            elif base_ok and not method_ok:
                regression += 1
            elif not base_ok and method_ok:
                if base_cat == "format_error":
                    format_thicket += 1
                else:
                    reasoning_thicket += 1
        result["strictly_correct"]  = strictly_correct
        result["reasoning_thicket"] = reasoning_thicket
        result["format_thicket"]    = format_thicket
        result["regression"]        = regression

    return result


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
    p.add_argument("--no-save",          action="store_true", default=True,
                   help="Disable checkpoint saving (default: True; use --save to enable)")
    p.add_argument("--save",             action="store_false", dest="no_save",
                   help="Enable checkpoint saving")
    p.add_argument("--perturb-verbose",  action="store_true", default=False,
                   help="Print per-layer progress during perturbation (default off on GPU)")
    # ES variant flags
    p.add_argument("--max-new-tokens",   type=int, default=4,
                   help="Max tokens to generate per example (default 4 for classification; use ~256 for generation tasks)")
    p.add_argument("--noise-type",       default="gaussian", choices=["gaussian", "rademacher"],
                   help="Perturbation noise distribution")
    p.add_argument("--one-sided",        action="store_true", default=False,
                   help="One-sided ES: only eval +eps (no antithetic -eps pair)")
    p.add_argument("--no-normalize",     action="store_true", default=False,
                   help="Disable z-score reward normalisation in gradient update")
    p.add_argument("--top-k",            type=int,   default=0,
                   help="ARS-style: keep only top-k seeds by |advantage| before update (0=all)")
    # prompt style
    p.add_argument("--prompt-style", default="default", choices=["default", "mezo"],
                   help="Prompt template style. 'mezo' uses exact templates from the MeZO paper "
                        "(2305.17333, Appendix Table 14). Note: SST-2 and CB use different label "
                        "words under 'mezo' (great/terrible and Yes/No/Maybe respectively).")
    # reward model
    p.add_argument("--reward", default="accuracy", choices=["accuracy", "ce"],
                   help="Training reward signal. 'accuracy': binary +1/-1 (default). "
                        "'ce': restricted log-softmax over label words — requires "
                        "task.label_words() to be non-None and white-box model access.")
    # decomposition tracking
    p.add_argument("--track-decomposition", action="store_true", default=False,
                   help="At each val step, run full per-example decomposition analysis")
    p.add_argument("--base-json",        type=str,   default=None,
                   help="Path to base model analyze_failures.py JSON for decomposition tracking")
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


def _prompt_fn(task, backend, prompt_style: str = "default"):
    """Select prompt builder based on prompt_style and backend capabilities."""
    if prompt_style == "mezo":
        return task.build_prompt_mezo
    if task.prefer_base_prompt:
        return task.build_prompt_base
    return task.build_prompt if backend.is_instruct else task.build_prompt_base


def _score_fn(task, prompt_style: str):
    """Select the score function matching the active prompt style."""
    return task.score_mezo if prompt_style == "mezo" else task.score


def _label_words(task, prompt_style: str):
    """Select label words matching the active prompt style."""
    return task.label_words_mezo() if prompt_style == "mezo" else task.label_words()


def eval_batch(backend, examples: list[dict], task, prompt_style: str = "default") -> float:
    """Evaluate a mini-batch using task.reward() as the training signal.

    For the default prompt style, delegates to task.reward() so tasks with
    custom composite rewards (e.g. countdown's format+answer signal) are
    preserved. For MeZO style, maps score_mezo > 0 to binary {0, 1} because
    score_mezo is always ±1 (no composite extension defined for MeZO).
    """
    prompt_fn = _prompt_fn(task, backend, prompt_style)
    raw = task.prefer_base_prompt or prompt_style == "mezo"
    prompts = [prompt_fn(ex) for ex in examples]
    outputs = backend.generate_batch(prompts, raw=raw)
    if prompt_style == "mezo":
        score_fn = task.score_mezo
        rewards = [1.0 if score_fn(text, ex) > 0 else 0.0 for text, ex in zip(outputs, examples)]
    else:
        rewards = [task.reward(text, ex) for text, ex in zip(outputs, examples)]
    return sum(rewards) / len(rewards)


def _score_ce_fn(task, prompt_style: str):
    """Select the CE score function matching the active prompt style."""
    return task.score_ce_mezo if prompt_style == "mezo" else task.score_ce


def eval_batch_ce(backend, examples: list[dict], task, prompt_style: str = "default") -> float:
    """Evaluate a mini-batch using CE reward (restricted log-softmax over label words).

    Single forward pass only — no generation needed. Returns mean log P(correct | prompt)
    over the batch, where the log-prob is restricted/normalized over label_words().
    Range: (-inf, 0], where 0 means the model is certain about every correct label.
    """
    prompt_fn = _prompt_fn(task, backend, prompt_style)
    raw = task.prefer_base_prompt or prompt_style == "mezo"
    prompts = [prompt_fn(ex) for ex in examples]
    lw = _label_words(task, prompt_style)
    log_probs_list = backend.score_logprobs_batch(prompts, lw, raw=raw)
    score_ce_fn = _score_ce_fn(task, prompt_style)
    rewards = [score_ce_fn(lp, ex) for lp, ex in zip(log_probs_list, examples)]
    return sum(rewards) / len(rewards)


def validate(backend, val_data: list[dict], task, batch_size: int = 16,
             prompt_style: str = "default", reward: str = "accuracy") -> float:
    """Validate over the full val set, chunked into batched forward passes.

    When reward='ce' and the task has label words, uses log-likelihood argmax for
    accuracy (matching the paper's evaluation strategy for classification tasks).
    Otherwise falls back to generation + regex scoring.
    """
    prompt_fn = _prompt_fn(task, backend, prompt_style)
    raw = task.prefer_base_prompt or prompt_style == "mezo"
    lw = _label_words(task, prompt_style)
    use_ce_eval = reward == "ce" and lw is not None

    correct = 0
    for i in range(0, len(val_data), batch_size):
        chunk = val_data[i:i + batch_size]
        prompts = [prompt_fn(ex) for ex in chunk]
        if use_ce_eval:
            log_probs_list = backend.score_logprobs_batch(prompts, lw, raw=raw)
            score_ce_fn = _score_ce_fn(task, prompt_style)
            for lp, ex in zip(log_probs_list, chunk):
                correct_lp = score_ce_fn(lp, ex)
                if correct_lp >= max(lp.values()) - 1e-9:
                    correct += 1
        else:
            score_fn = _score_fn(task, prompt_style)
            outputs = backend.generate_batch(prompts, raw=raw)
            correct += sum(score_fn(text, ex) > 0 for text, ex in zip(outputs, chunk))
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

    ps = args.prompt_style
    _eval = (lambda b, ex, t: eval_batch_ce(b, ex, t, ps)) if args.reward == "ce" \
            else (lambda b, ex, t: eval_batch(b, ex, t, ps))
    nt = args.noise_type
    for _ in range(args.population_size):
        seed = random.randint(0, 2**31)

        perturb_inplace(model, seed, args.sigma, sign=+1, noise_type=nt)  # θ → θ+σε
        r_plus = _eval(backend, batch, task)
        fwd_passes += effective_batch

        if args.one_sided:
            restore_inplace(model, seed, args.sigma, sign=+1, noise_type=nt)  # θ+σε → θ
            adv = r_plus
        else:
            perturb_inplace(model, seed, 2 * args.sigma, sign=-1, noise_type=nt)  # θ+σε → θ-σε
            r_minus = _eval(backend, batch, task)
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
    decomp_log_path = run_dir / "decomp_log.jsonl"
    log_entry(log_path, {"event": "config", **vars(args), "resolved_dtype": dtype, "normalize": normalize})
    print(f"Run dir: {run_dir}")

    # Load base model categories for decomposition tracking
    base_categories = None
    if args.track_decomposition:
        if args.base_json is None:
            raise SystemExit("[error] --track-decomposition requires --base-json")
        base_json = json.loads(Path(args.base_json).read_text())
        base_categories = [ex["category"] for ex in base_json["examples"]]
        print(f"[decomp] loaded base categories from {args.base_json} (n={len(base_categories)})")

    task = get_task(args.task)
    if args.reward == "ce" and task.label_words() is None:
        raise SystemExit(
            f"[error] --reward ce is not supported for task '{args.task}': "
            f"task.label_words() returned None. Use --reward accuracy instead."
        )

    t_start = time.perf_counter()
    backend = create_backend(
        "hf",
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

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
        f"reward={args.reward}  train={len(train_data)}  val={len(val_data)}\n"
        f"no_save={args.no_save}  prompt_style={args.prompt_style}\n"
        f"seed={args.seed}  early_stop_delta={args.early_stop_delta}  val_every={args.val_every}"
    )

    baseline_acc = validate(backend, val_data, task, batch_size=args.batch_size, prompt_style=args.prompt_style, reward=args.reward)
    print(f"Baseline val_acc: {baseline_acc:.3f} ({int(baseline_acc * len(val_data))}/{len(val_data)})")
    log_entry(log_path, {"event": "baseline", "val_acc": baseline_acc})

    best_val_acc = run_state.get("best_val_acc", baseline_acc)
    start_iter = run_state.get("iteration", 0)
    # train_fwd: forward passes spent on training perturbations only (the controlled budget).
    # total_fwd: train_fwd + validation passes (informational; not used as x-axis).
    cumulative_train_fwd = run_state.get("cumulative_train_fwd", 0)
    cumulative_total_fwd = run_state.get("cumulative_total_fwd", 0)
    if not args.no_save:
        save_checkpoint(backend.model, run_dir / "best.pt", {
            "iteration": start_iter, "best_val_acc": best_val_acc,
            "cumulative_train_fwd": cumulative_train_fwd,
            "cumulative_total_fwd": cumulative_total_fwd,
        })

    for iteration in range(args.num_iters):
        t_iter = time.perf_counter()

        seeds, advantages, iter_fwd = run_es_iteration(backend.model, backend, task, train_data, args)
        # MeZO (SPSA) requires dividing by 2σ to form an unbiased gradient estimate:
        #   ĝ = (r⁺ − r⁻) / (2σ) · ε  (trainer.py:780 in the reference implementation)
        # Other ES variants use z-score normalized advantages (dimensionless), so σ
        # scaling is already absorbed and must NOT be applied a second time.
        effective_lr = args.lr / (2 * args.sigma) if args.prompt_style == "mezo" else args.lr
        es_grad_update(
            backend.model, seeds, advantages,
            lr=effective_lr,
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
        if not args.no_save:
            save_checkpoint(backend.model, run_dir / "latest.pt", current_state)

        if (iteration + 1) % args.val_every == 0:
            if args.track_decomposition:
                decomp = decomp_eval(backend, val_data, task, args.batch_size, base_categories, prompt_style=args.prompt_style)
                val_acc = decomp["val_acc"]
                cumulative_total_fwd += len(val_data)
                decomp_entry = {
                    "iteration": start_iter + iteration + 1,
                    "train_fwd": cumulative_train_fwd,
                    **{k: v for k, v in decomp.items() if k != "categories"},
                }
                log_entry(decomp_log_path, decomp_entry)
                decomp_summary = (
                    f"strict={decomp.get('strictly_correct', '?')} "
                    f"reason={decomp.get('reasoning_thicket', '?')} "
                    f"format={decomp.get('format_thicket', '?')} "
                    f"regress={decomp.get('regression', '?')}"
                )
                print(f"  [decomp] {decomp_summary}")
            else:
                val_acc = validate(backend, val_data, task, batch_size=args.batch_size, prompt_style=args.prompt_style, reward=args.reward)
                cumulative_total_fwd += len(val_data)

            entry["val_acc"] = val_acc
            entry["total_fwd"] = cumulative_total_fwd

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if not args.no_save:
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
