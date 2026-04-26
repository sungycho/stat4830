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
from src.tasks import get_task, available_tasks, PROMPT_STYLES, resolve_prompt_config, PromptConfig
from src.utils.seeds import set_seeds
from src.utils.perturb import perturb_inplace, restore_inplace, es_grad_update

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b",  re.IGNORECASE)
_MAX_PARSE_FAIL_SAMPLES = 10


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
                prompt_fn, raw: bool) -> dict:
    """Run full per-example decomposition eval over val_data.

    Returns aggregate counts and per-example categories.
    base_categories: list of 'correct'/'wrong_answer'/'format_error' from base model run,
                     indexed identically to val_data. If None, skips decomposition counts.
    """
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
    p.add_argument("--lr",               type=float, default=1e-4,
                   help="Learning rate passed directly to es_grad_update. "
                        "All constants (2σ, population size, etc.) are absorbed here.")
    p.add_argument("--train-size",       type=int,   default=64)
    p.add_argument("--val-size",         type=int,   default=200)
    p.add_argument("--early-stop-delta", type=float, default=0.0,
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
    p.add_argument("--prompt-style", default="simple", choices=PROMPT_STYLES,
                   help="Prompt template style. 'simple': few-shot completion (build_prompt_base). "
                        "'complex': instruction format (build_prompt). "
                        "'mezo': MeZO paper templates (2305.17333, Table 14), always raw. "
                        "'free': bare input, no examples or instructions.")
    p.add_argument("--chat-template", dest="chat_template", action="store_true", default=None,
                   help="Force apply the model's chat template (overrides auto-detect).")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false",
                   help="Force skip the chat template regardless of model type.")
    p.set_defaults(chat_template=None)
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
        v, internalstate, gauss = random.getstate()
        state_path = path.with_suffix(".state.json")
        with open(state_path, "w") as f:
            json.dump({
                **run_state,
                "_rng_torch_cpu": torch.get_rng_state().tolist(),
                "_rng_python_version": v,
                "_rng_python_state": list(internalstate),
                "_rng_python_gauss": gauss,
            }, f)


def load_run_state(checkpoint_path: str) -> dict:
    """Load saved run state (iteration, best_val_acc) alongside a checkpoint."""
    state_path = Path(checkpoint_path).with_suffix(".state.json")
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def _compute_raw(force_raw: bool, chat_template_override, backend) -> bool:
    """Determine whether to skip the chat template when calling the backend.

    force_raw=True (mezo) always bypasses the template.
    chat_template_override=True/False lets the user force a specific behaviour.
    None auto-detects: skip template if the tokenizer has none.
    """
    if force_raw:
        return True
    if chat_template_override is not None:
        return not chat_template_override
    return not bool(getattr(backend.tokenizer, "chat_template", None))


def eval_batch(backend, examples: list[dict], task, prompt_cfg: PromptConfig, raw: bool) -> float:
    """Evaluate a mini-batch using task.reward() as the training signal.

    For mezo (force_raw=True), maps score_mezo > 0 to binary {0, 1}.
    For other styles, delegates to task.reward() so tasks with custom composite
    rewards (e.g. countdown's format+answer signal) are preserved.
    """
    prompts = [prompt_cfg.prompt_fn(ex) for ex in examples]
    outputs = backend.generate_batch(prompts, raw=raw)
    if prompt_cfg.force_raw:
        rewards = [1.0 if prompt_cfg.score_fn(text, ex) > 0 else 0.0 for text, ex in zip(outputs, examples)]
    else:
        rewards = [task.reward(text, ex) for text, ex in zip(outputs, examples)]
    return sum(rewards) / len(rewards)


def eval_batch_ce(backend, examples: list[dict], task, prompt_cfg: PromptConfig, raw: bool) -> float:
    """Evaluate a mini-batch using CE reward (restricted log-softmax over label words).

    Single forward pass only — no generation needed. Returns mean log P(correct | prompt)
    over the batch, where the log-prob is restricted/normalized over label_words().
    Range: (-inf, 0], where 0 means the model is certain about every correct label.
    """
    prompts = [prompt_cfg.prompt_fn(ex) for ex in examples]
    log_probs_list = backend.score_logprobs_batch(prompts, prompt_cfg.label_words, raw=raw)
    rewards = [prompt_cfg.score_ce_fn(lp, ex) for lp, ex in zip(log_probs_list, examples)]
    return sum(rewards) / len(rewards)


def _gold_word_for_ce(label_words: list[str], score_ce_fn, example: dict) -> str:
    """Find the gold label word for a CE example without knowing the task's internal mapping.

    Probes score_ce_fn with a dummy log_probs where each word has its index as value.
    score_ce_fn returns log_probs[correct_word], so the result is the correct word's index.
    """
    dummy = {w: float(i) for i, w in enumerate(label_words)}
    return label_words[int(round(score_ce_fn(dummy, example)))]


def validate_with_dist(backend, val_data: list[dict], task, batch_size: int,
                       prompt_cfg: PromptConfig, raw: bool, reward: str = "accuracy") -> dict:
    """Validate over the full val set and collect per-gold-label prediction distributions.

    Returns {"val_acc": float, "pred_by_gold": {gold: {pred: count}}, "gold_dist": {gold: count}}.

    CE path: pred_by_gold uses label_words space (e.g. "great"/"terrible" for SST-2 mezo).
             No parse_fail since argmax always yields a valid word.
    Accuracy path: pred_by_gold uses task.predict()/gold_label() natural label strings.
                   None from predict() maps to "parse_fail". val_acc still uses score_fn.
    """
    from collections import Counter
    use_ce_eval = reward == "ce" and prompt_cfg.label_words is not None

    gold_dist: Counter = Counter()
    pred_by_gold: dict[str, Counter] = {}
    parse_fail_samples: list[dict] = []
    correct = 0

    for i in range(0, len(val_data), batch_size):
        chunk = val_data[i:i + batch_size]
        prompts = [prompt_cfg.prompt_fn(ex) for ex in chunk]
        if use_ce_eval:
            log_probs_list = backend.score_logprobs_batch(prompts, prompt_cfg.label_words, raw=raw)
            for lp, ex in zip(log_probs_list, chunk):
                gold = _gold_word_for_ce(prompt_cfg.label_words, prompt_cfg.score_ce_fn, ex)
                pred = max(lp, key=lp.get)
                correct_lp = prompt_cfg.score_ce_fn(lp, ex)
                if correct_lp >= max(lp.values()) - 1e-9:
                    correct += 1
                gold_dist[gold] += 1
                pred_by_gold.setdefault(gold, Counter())[pred] += 1
        else:
            outputs = backend.generate_batch(prompts, raw=raw)
            lw = task.label_words()  # None for generation tasks (GSM8K, etc.)
            for text, ex, prompt in zip(outputs, chunk, prompts):
                gold = task.gold_label(ex)
                pred = task.predict(text)
                score_val = prompt_cfg.score_fn(text, ex)
                if score_val > 0:
                    correct += 1
                # Distribution label: classification → specific label string;
                # generation → correct / wrong_number / parse_fail
                if pred is None:
                    dist_label = "parse_fail"
                elif lw is not None:
                    dist_label = pred
                else:
                    dist_label = "correct" if score_val > 0 else "wrong_number"
                gold_dist[gold] += 1
                pred_by_gold.setdefault(gold, Counter())[dist_label] += 1
                if pred is None and len(parse_fail_samples) < _MAX_PARSE_FAIL_SAMPLES:
                    parse_fail_samples.append({"gold": gold, "generated": repr(text), "prompt_tail": prompt[-120:]})

    return {
        "val_acc": correct / len(val_data),
        "gold_dist": dict(gold_dist),
        "pred_by_gold": {g: dict(c) for g, c in pred_by_gold.items()},
        "parse_fail_samples": parse_fail_samples,
    }


def validate(backend, val_data: list[dict], task, batch_size: int,
             prompt_cfg: PromptConfig, raw: bool, reward: str = "accuracy") -> float:
    return validate_with_dist(backend, val_data, task, batch_size, prompt_cfg, raw, reward)["val_acc"]


def run_es_iteration(
    model, backend, task, train_data, args, prompt_cfg: PromptConfig, raw: bool
) -> tuple[list[int], list[float], int]:
    """Run one ES iteration. Returns (seeds, advantages, fwd_passes_used)."""
    effective_batch = min(args.batch_size, len(train_data))
    if effective_batch < args.batch_size:
        print(f"[warn] batch_size={args.batch_size} > train_size={len(train_data)}, clamped to {effective_batch}")
    batch = random.sample(train_data, effective_batch)
    seeds, advantages = [], []
    fwd_passes = 0

    _eval = (lambda b, ex, t: eval_batch_ce(b, ex, t, prompt_cfg, raw)) if args.reward == "ce" \
            else (lambda b, ex, t: eval_batch(b, ex, t, prompt_cfg, raw))
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

    prompt_cfg = resolve_prompt_config(task, args.prompt_style)
    raw = _compute_raw(prompt_cfg.force_raw or task.prefer_base_prompt, args.chat_template, backend)

    run_state = {}
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=True)
        backend.model.load_state_dict(ckpt)
        run_state = load_run_state(args.resume_from)
        if run_state:
            if "_rng_torch_cpu" in run_state:
                torch.set_rng_state(torch.tensor(run_state["_rng_torch_cpu"], dtype=torch.uint8))
                random.setstate((
                    run_state["_rng_python_version"],
                    tuple(run_state["_rng_python_state"]),
                    run_state["_rng_python_gauss"],
                ))
                print(f"Resumed from: {args.resume_from} | iter={run_state.get('iteration', 0)} best_val={run_state.get('best_val_acc', 'n/a')} | RNG restored")
            else:
                print(f"Resumed from: {args.resume_from} | iter={run_state.get('iteration', 0)} best_val={run_state.get('best_val_acc', 'n/a')} | [warn] no RNG state in checkpoint")
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
        f"no_save={args.no_save}  prompt_style={args.prompt_style}  "
        f"chat_template={'auto-detect' if args.chat_template is None else args.chat_template} → active={not raw}\n"
        f"seed={args.seed}  early_stop_delta={args.early_stop_delta}  val_every={args.val_every}"
    )
    _prompt_cfg_tmp = resolve_prompt_config(task, args.prompt_style)
    use_ce_eval = args.reward == "ce" and _prompt_cfg_tmp.label_words is not None
    if use_ce_eval:
        _dist_labels = "/".join(_prompt_cfg_tmp.label_words)
        print(f"[pred_dist] tracking ON — labels: {_dist_labels} (CE argmax)")
    else:
        _dist_labels = "/".join(task.label_words() or ["?"])
        print(f"[pred_dist] tracking ON — labels: {_dist_labels} + parse_fail (from task.predict)")

    baseline_result = validate_with_dist(backend, val_data, task, args.batch_size, prompt_cfg, raw, reward=args.reward)
    baseline_acc = baseline_result["val_acc"]
    print(f"Baseline val_acc: {baseline_acc:.3f} ({int(baseline_acc * len(val_data))}/{len(val_data)})")
    baseline_log = {"event": "baseline", "val_acc": baseline_acc,
                    "pred_by_gold": baseline_result["pred_by_gold"],
                    "gold_dist": baseline_result["gold_dist"]}
    if baseline_result["parse_fail_samples"]:
        baseline_log["parse_fail_samples"] = baseline_result["parse_fail_samples"]
    log_entry(log_path, baseline_log)

    best_val_acc = run_state.get("best_val_acc", baseline_acc)
    start_iter = run_state.get("iteration", 0)
    # train_fwd: forward passes spent on training perturbations only (the controlled budget).
    # total_fwd: train_fwd + validation passes (informational; not used as x-axis).
    cumulative_train_fwd = run_state.get("cumulative_train_fwd", 0)
    cumulative_total_fwd = run_state.get("cumulative_total_fwd", 0)
    if not args.no_save and not run_state:
        save_checkpoint(backend.model, run_dir / "best.pt", {
            "iteration": start_iter, "best_val_acc": best_val_acc,
            "cumulative_train_fwd": cumulative_train_fwd,
            "cumulative_total_fwd": cumulative_total_fwd,
        })

    for iteration in range(args.num_iters):
        t_iter = time.perf_counter()

        seeds, advantages, iter_fwd = run_es_iteration(backend.model, backend, task, train_data, args, prompt_cfg, raw)
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
        if not args.no_save:
            save_checkpoint(backend.model, run_dir / "latest.pt", current_state)

        if (iteration + 1) % args.val_every == 0:
            if args.track_decomposition:
                decomp = decomp_eval(backend, val_data, task, args.batch_size, base_categories, prompt_cfg.prompt_fn, raw)
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
                val_result = validate_with_dist(backend, val_data, task, args.batch_size, prompt_cfg, raw, reward=args.reward)
                val_acc = val_result["val_acc"]
                cumulative_total_fwd += len(val_data)
                entry["pred_by_gold"] = val_result["pred_by_gold"]
                entry["gold_dist"] = val_result["gold_dist"]
                if val_result["parse_fail_samples"]:
                    entry["parse_fail_samples"] = val_result["parse_fail_samples"]

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

    try:
        from src.scripts.plot_pred_dist_evolution import plot_evolution
        plot_evolution(log_path, out_path=run_dir / "pred_dist_evolution.png")
    except Exception as e:
        print(f"[warn] Could not generate pred_dist_evolution plot: {e}")

    print(f"\nDone. Total: {time.perf_counter() - t_start:.1f}s | logs → {log_path}")


if __name__ == "__main__":
    main()
