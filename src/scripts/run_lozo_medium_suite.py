"""Orchestrate LOZO medium-model suites with reproducibility guards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ["SST-2", "sst-5", "SNLI", "MNLI", "RTE", "trec"]
DEFAULT_SEEDS = [13, 21, 42, 87, 100]
DEFAULT_K_VALUES = [16, 512]
DEBUG_LOG_PATH = Path(
    "/Users/gyubin/Documents/Git/stat4830/.cursor/debug-a13946.log"
)
DEBUG_SESSION_ID = "a13946"


def _debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        return


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_text(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _format_num(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _safe_read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def _tail_lines(path: Path, n: int) -> list[str]:
    lines = _safe_read_lines(path)
    if not lines:
        return []
    return lines[-n:]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    script: str
    batch_size: int
    lr: float
    eps: float
    step: int
    eval_step: int
    wd: float = 0.0
    extra_tag: str = "ft"
    rank: int = 4
    step_interval: int = 100
    lozo_optimizer: str = "sgd"
    beta1: float = 0.9


@dataclass(frozen=True)
class RunSpec:
    task: str
    k: int
    seed: int
    method: str
    model: str
    model_name: str
    script: str
    env: dict[str, str]
    command: list[str]
    data_dir: str
    result_dir: str
    completion_glob: str
    log_path: str


def _mezo_gr_tag(cfg: MethodConfig, seed: int) -> str:
    return (
        f"seed{seed}-bs{cfg.batch_size}-lr{_format_num(cfg.lr)}"
        f"-eps{_format_num(cfg.eps)}"
        f"-wd{_format_num(cfg.wd)}-step{cfg.step}-evalstep{cfg.eval_step}"
    )


def _lozo_gr_tag(cfg: MethodConfig, seed: int) -> str:
    return (
        f"seed{seed}-bs{cfg.batch_size}-lr{_format_num(cfg.lr)}"
        f"-eps{_format_num(cfg.eps)}"
        f"-wd{_format_num(cfg.wd)}-step{cfg.step}-evalstep{cfg.eval_step}"
        f"-step-interval{cfg.step_interval}-rank{cfg.rank}"
    )


def _mezo_tag(cfg: MethodConfig, k: int, model_name: str) -> str:
    return f"k{k}-{model_name}-mezo-{cfg.extra_tag}"


def _lozo_tag(cfg: MethodConfig, k: int, model_name: str) -> str:
    return (
        f"k{k}-{model_name}-lowrank-{cfg.extra_tag}-{cfg.lozo_optimizer}"
        f"-beta1-{_format_num(cfg.beta1)}"
    )


def _completion_glob(task: str) -> str:
    return f"test_results_{task}.txt"


def _expected_paths(
    medium_models_dir: Path,
    task: str,
    k: int,
    seed: int,
    model_name: str,
    cfg: MethodConfig,
) -> tuple[Path, Path]:
    if cfg.name == "mezo":
        gr_tag = _mezo_gr_tag(cfg, seed)
        tag = _mezo_tag(cfg, k, model_name)
    elif cfg.name == "lozo":
        gr_tag = _lozo_gr_tag(cfg, seed)
        tag = _lozo_tag(cfg, k, model_name)
    else:
        raise ValueError(f"Unknown method: {cfg.name}")

    result_dir = (
        medium_models_dir
        / "result"
        / f"{task}-{model_name}-prompt-standard-{tag}-{gr_tag}"
        / f"{k}-{seed}"
    )
    log_path = (
        medium_models_dir / "log_dir" / f"{task}-{gr_tag}-{tag}.log"
    )
    return result_dir, log_path


def _build_run_spec(
    medium_models_dir: Path,
    task: str,
    k: int,
    seed: int,
    model: str,
    model_name: str,
    cfg: MethodConfig,
) -> RunSpec:
    result_dir, log_path = _expected_paths(
        medium_models_dir=medium_models_dir,
        task=task,
        k=k,
        seed=seed,
        model_name=model_name,
        cfg=cfg,
    )
    env = {
        "TASK": task,
        "K": str(k),
        "SEED": str(seed),
        "BS": str(cfg.batch_size),
        "LR": _format_num(cfg.lr),
        "EPS": _format_num(cfg.eps),
        "WD": _format_num(cfg.wd),
        "STEP": str(cfg.step),
        "EVAL_STEP": str(cfg.eval_step),
        "MODEL": model,
        "MODELNAME": model_name,
        "EXTRA_TAG": cfg.extra_tag,
    }
    if cfg.name == "lozo":
        env.update(
            {
                "RANK": str(cfg.rank),
                "STEP_INTERVAL": str(cfg.step_interval),
                "LOZO_OPTIMIZER": cfg.lozo_optimizer,
                "BETA1": _format_num(cfg.beta1),
            }
        )
    return RunSpec(
        task=task,
        k=k,
        seed=seed,
        method=cfg.name,
        model=model,
        model_name=model_name,
        script=cfg.script,
        env=env,
        command=["bash", cfg.script],
        data_dir=str(
            medium_models_dir
            / "data"
            / "k-shot-1k-test"
            / task
            / f"{k}-{seed}"
        ),
        result_dir=str(result_dir),
        completion_glob=_completion_glob(task),
        log_path=str(log_path),
    )


def _is_completed(spec: RunSpec) -> bool:
    marker = Path(spec.result_dir) / spec.completion_glob
    return marker.exists()


def _extract_metrics(spec: RunSpec) -> dict[str, float]:
    marker = Path(spec.result_dir) / spec.completion_glob
    metrics: dict[str, float] = {}
    for line in _safe_read_lines(marker):
        if "=" not in line:
            continue
        key, val = [p.strip() for p in line.split("=", 1)]
        try:
            metrics[key] = float(val)
        except ValueError:
            continue
    return metrics


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _resolve_lozo_sha(lozo_root: Path) -> str | None:
    if not (lozo_root / ".git").exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(lozo_root), "rev-parse", "HEAD"],
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _assert_binaries() -> None:
    required = ["bash", "git", "jq"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise EnvironmentError(
            "Missing required binaries: "
            + ", ".join(missing)
            + ". Install OS dependencies and retry."
        )


def _parse_major(version: str) -> int | None:
    head = version.strip().split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _assert_transformers_compat() -> None:
    try:
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise EnvironmentError(
            "Could not import `transformers`; run `uv sync` and retry."
        ) from exc
    version = str(getattr(transformers, "__version__", ""))
    major = _parse_major(version)
    if major is not None and major >= 5:
        raise EnvironmentError(
            "Incompatible transformers version detected: "
            f"{version}. LOZO medium requires transformers<5. "
            "Run `uv sync` after pulling latest repo changes."
        )


def _assert_medium_models_layout(medium_models_dir: Path) -> None:
    required = [
        medium_models_dir / "lozo.sh",
        medium_models_dir / "mezo.sh",
        medium_models_dir / "run_lozo.py",
        medium_models_dir / "run_mezo.py",
        medium_models_dir / "run_fewshot_lozo.sh",
        medium_models_dir / "run_fewshot.sh",
        medium_models_dir / "src" / "LOZOtrainer.py",
        medium_models_dir / "src" / "trainer.py",
        medium_models_dir / "src" / "modeling_roberta.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "LOZO medium_models layout is incomplete. Missing files:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def _assert_writable(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise PermissionError(f"{label} is not writable: {path}")


def _validate_model_config(model: str, require_local_model: bool) -> None:
    raw = model.strip()
    if not raw:
        raise ValueError("Model argument cannot be empty.")
    model_path = Path(raw).expanduser()
    if require_local_model and not model_path.exists():
        raise FileNotFoundError(
            "--require-local-model was set, but model path "
            f"does not exist: {model_path}"
        )


def _validate_run_specs(
    run_specs: list[RunSpec],
    medium_models_dir: Path,
    skip_data_check: bool,
) -> None:
    _assert_writable(medium_models_dir / "log_dir", "LOZO log directory")
    _assert_writable(medium_models_dir / "result", "LOZO result directory")
    if skip_data_check:
        return
    missing_data = [
        Path(spec.data_dir)
        for spec in run_specs
        if not Path(spec.data_dir).exists()
    ]
    if missing_data:
        examples = "\n".join(f"  - {p}" for p in missing_data[:10])
        raise FileNotFoundError(
            "Missing expected k-shot dataset directories. "
            "Generate datasets first with LOZO tools.\n"
            f"Examples:\n{examples}"
        )


def _gpu_memory_snapshot_mb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None
    vals: list[float] = []
    for line in out.splitlines():
        try:
            vals.append(float(line.strip()))
        except ValueError:
            continue
    if not vals:
        return None
    return max(vals)


def _infer_failure_hints(log_tail: list[str], return_code: int) -> list[str]:
    hints: list[str] = []
    text = "\n".join(log_tail)
    if "ModuleNotFoundError: No module named 'loralib'" in text:
        hints.append(
            "Missing Python dependency `loralib`; run `uv sync` and retry."
        )
    if "ModuleNotFoundError: No module named 'loguru'" in text:
        hints.append(
            "Missing Python dependency `loguru`; run `uv sync` and retry."
        )
    if "ModuleNotFoundError: No module named 'sklearn'" in text:
        hints.append(
            "Missing Python dependency `scikit-learn`; "
            "run `uv sync` and retry."
        )
    if (
        "cannot import name 'find_pruneable_heads_and_indices'" in text
        and "transformers.pytorch_utils" in text
    ):
        hints.append(
            "Transformers major-version mismatch detected. "
            "Run `uv sync` to install repo-pinned transformers<5."
        )
    if "No such file or directory" in text and "k-shot-1k-test" in text:
        hints.append(
            "Missing k-shot dataset directories; "
            "run LOZO data generation first."
        )
    if "CUDA out of memory" in text:
        hints.append("CUDA OOM; lower batch size or use a smaller model.")
    if return_code != 0 and not hints:
        hints.append(
            "Run failed; inspect `log_path` tail for traceback details."
        )
    return hints


def build_method_configs(args: argparse.Namespace) -> dict[str, MethodConfig]:
    profile = args.profile
    if profile == "smoke":
        mezo = MethodConfig(
            name="mezo",
            script="mezo.sh",
            batch_size=args.mezo_bs,
            lr=args.mezo_lr,
            eps=args.mezo_eps,
            step=args.smoke_steps,
            eval_step=args.smoke_eval_steps,
            wd=args.weight_decay,
        )
        lozo = MethodConfig(
            name="lozo",
            script="lozo.sh",
            batch_size=args.lozo_bs,
            lr=args.lozo_lr,
            eps=args.lozo_eps,
            step=args.smoke_steps,
            eval_step=args.smoke_eval_steps,
            wd=args.weight_decay,
            rank=args.lozo_rank,
            step_interval=args.lozo_step_interval,
            lozo_optimizer=args.lozo_optimizer,
            beta1=args.lozo_beta1,
        )
    else:
        mezo = MethodConfig(
            name="mezo",
            script="mezo.sh",
            batch_size=args.mezo_bs,
            lr=args.mezo_lr,
            eps=args.mezo_eps,
            step=args.full_steps,
            eval_step=args.full_eval_steps,
            wd=args.weight_decay,
        )
        lozo = MethodConfig(
            name="lozo",
            script="lozo.sh",
            batch_size=args.lozo_bs,
            lr=args.lozo_lr,
            eps=args.lozo_eps,
            step=args.full_steps,
            eval_step=args.full_eval_steps,
            wd=args.weight_decay,
            rank=args.lozo_rank,
            step_interval=args.lozo_step_interval,
            lozo_optimizer=args.lozo_optimizer,
            beta1=args.lozo_beta1,
        )
    return {"mezo": mezo, "lozo": lozo}


def default_axes_for_profile(
    profile: str,
) -> tuple[list[str], list[int], list[int]]:
    if profile == "smoke":
        return (["SST-2", "RTE"], [16], [42])
    return (DEFAULT_TASKS, DEFAULT_K_VALUES, DEFAULT_SEEDS)


def build_run_matrix(
    args: argparse.Namespace,
    medium_models_dir: Path,
) -> list[RunSpec]:
    if args.tasks:
        tasks = _parse_csv_text(args.tasks)
    else:
        tasks, _, _ = default_axes_for_profile(args.profile)
    if args.k_values:
        k_values = _parse_csv_ints(args.k_values)
    else:
        _, k_values, _ = default_axes_for_profile(args.profile)
    if args.seeds:
        seeds = _parse_csv_ints(args.seeds)
    else:
        _, _, seeds = default_axes_for_profile(args.profile)

    methods = _parse_csv_text(args.methods)
    method_configs = build_method_configs(args)
    unknown = [m for m in methods if m not in method_configs]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    run_specs: list[RunSpec] = []
    for task in tasks:
        for k in k_values:
            for seed in seeds:
                for method in methods:
                    run_specs.append(
                        _build_run_spec(
                            medium_models_dir=medium_models_dir,
                            task=task,
                            k=k,
                            seed=seed,
                            model=args.model,
                            model_name=args.model_name,
                            cfg=method_configs[method],
                        )
                    )
    if args.max_runs is not None:
        run_specs = run_specs[: args.max_runs]
    return run_specs


def _run_one(
    spec: RunSpec,
    medium_models_dir: Path,
    dry_run: bool,
    poll_gpu_memory_sec: float,
    log_tail_lines: int,
) -> tuple[str, dict[str, Any]]:
    env = os.environ.copy()
    env.update(spec.env)
    run_id = (
        f"{spec.method}:{spec.task}:"
        f"k{spec.k}:seed{spec.seed}:{int(time.time())}"
    )
    resolved_python = shutil.which("python", path=env.get("PATH"))
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H5",
        location="src/scripts/run_lozo_medium_suite.py:_run_one:pre_run",
        message="Resolved python for upstream shell run",
        data={
            "cwd": str(medium_models_dir),
            "command": spec.command,
            "resolved_python": resolved_python,
            "path_head": env.get("PATH", "").split(":")[:5],
        },
    )
    # endregion
    started_at = time.time()
    payload: dict[str, Any] = {
        "task": spec.task,
        "k": spec.k,
        "seed": spec.seed,
        "method": spec.method,
        "script": spec.script,
        "command": spec.command,
        "env": spec.env,
        "data_dir": spec.data_dir,
        "result_dir": spec.result_dir,
        "log_path": spec.log_path,
        "started_at": started_at,
        "gpu_memory_mb_before": _gpu_memory_snapshot_mb(),
    }
    if dry_run:
        payload["status"] = "dry_run"
        payload["finished_at"] = time.time()
        payload["elapsed_sec"] = payload["finished_at"] - started_at
        return "dry_run", payload

    proc = subprocess.Popen(
        spec.command,
        cwd=medium_models_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    peak_mem = payload["gpu_memory_mb_before"]
    while proc.poll() is None:
        snap = _gpu_memory_snapshot_mb()
        if snap is not None:
            peak_mem = snap if peak_mem is None else max(peak_mem, snap)
        time.sleep(max(0.2, poll_gpu_memory_sec))

    payload["return_code"] = proc.returncode
    payload["finished_at"] = time.time()
    payload["elapsed_sec"] = payload["finished_at"] - started_at
    payload["gpu_memory_mb_peak"] = peak_mem
    payload["gpu_memory_mb_after"] = _gpu_memory_snapshot_mb()
    log_tail = _tail_lines(Path(spec.log_path), log_tail_lines)
    payload["log_tail"] = log_tail
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H1_H2_H3_H4",
        location="src/scripts/run_lozo_medium_suite.py:_run_one:post_run",
        message="Upstream run completed",
        data={
            "return_code": proc.returncode,
            "elapsed_sec": payload["elapsed_sec"],
            "log_path": spec.log_path,
            "log_tail": log_tail,
        },
    )
    # endregion

    if proc.returncode == 0 and _is_completed(spec):
        payload["status"] = "completed"
        payload["metrics"] = _extract_metrics(spec)
        return "completed", payload
    if proc.returncode == 0 and not _is_completed(spec):
        payload["status"] = "no_marker"
        payload["failure_hints"] = _infer_failure_hints(
            log_tail,
            proc.returncode,
        )
        return "no_marker", payload
    payload["status"] = "failed"
    payload["failure_hints"] = _infer_failure_hints(log_tail, proc.returncode)
    return "failed", payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run LOZO medium-model suite in a reproducible, resumable way."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lozo-root",
        default="external/LOZO",
        help="Path to cloned LOZO repo root",
    )
    p.add_argument(
        "--run-root",
        default=f"results/lozo_medium_suite_{_stamp()}",
        help="Where to write manifests and run logs for this orchestration",
    )
    p.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    p.add_argument(
        "--tasks",
        default="",
        help="CSV tasks (e.g. SST-2,sst-5,SNLI)",
    )
    p.add_argument("--k-values", default="", help="CSV k values")
    p.add_argument("--seeds", default="", help="CSV seeds")
    p.add_argument("--methods", default="mezo,lozo", help="CSV methods")
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap number of runs",
    )
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--model-name", default="roberta-large")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--mezo-bs", type=int, default=64)
    p.add_argument("--mezo-lr", type=float, default=1e-6)
    p.add_argument("--mezo-eps", type=float, default=1e-3)
    p.add_argument("--lozo-bs", type=int, default=64)
    p.add_argument("--lozo-lr", type=float, default=1e-7)
    p.add_argument("--lozo-eps", type=float, default=1e-3)
    p.add_argument("--lozo-rank", type=int, default=4)
    p.add_argument("--lozo-step-interval", type=int, default=100)
    p.add_argument("--lozo-optimizer", default="sgd")
    p.add_argument("--lozo-beta1", type=float, default=0.9)
    p.add_argument("--full-steps", type=int, default=100000)
    p.add_argument("--full-eval-steps", type=int, default=10000)
    p.add_argument("--smoke-steps", type=int, default=1000)
    p.add_argument("--smoke-eval-steps", type=int, default=100)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if completion marker exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests without launching runs",
    )
    p.add_argument(
        "--expected-lozo-ref",
        default="",
        help="If set, fail unless LOZO HEAD matches this commit hash.",
    )
    p.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip fail-fast checks for expected k-shot dataset directories.",
    )
    p.add_argument(
        "--require-local-model",
        action="store_true",
        help="Require --model to resolve to an existing local path.",
    )
    p.add_argument(
        "--poll-gpu-memory-sec",
        type=float,
        default=2.0,
        help="Sampling interval in seconds for GPU memory snapshots.",
    )
    p.add_argument(
        "--log-tail-lines",
        type=int,
        default=40,
        help=(
            "Attach this many trailing lines from upstream run log "
            "to each record."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lozo_root = Path(args.lozo_root).expanduser().resolve()
    medium_models_dir = lozo_root / "medium_models"
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # region agent log
    _debug_log(
        run_id=f"main:{int(time.time())}",
        hypothesis_id="H1_H2_H3_H4_H5",
        location="src/scripts/run_lozo_medium_suite.py:main:start",
        message="Runner invoked",
        data={
            "profile": args.profile,
            "methods": args.methods,
            "model": args.model,
            "lozo_root": args.lozo_root,
            "skip_data_check": args.skip_data_check,
        },
    )
    # endregion

    _assert_binaries()
    _assert_transformers_compat()
    _validate_model_config(args.model, args.require_local_model)
    if not medium_models_dir.exists():
        raise FileNotFoundError(
            f"Could not find medium_models at {medium_models_dir}. "
            "Clone LOZO first (e.g. git clone "
            "https://github.com/optsuite/LOZO external/LOZO)."
        )
    _assert_medium_models_layout(medium_models_dir)

    run_specs = build_run_matrix(args, medium_models_dir)
    _validate_run_specs(
        run_specs=run_specs,
        medium_models_dir=medium_models_dir,
        skip_data_check=args.skip_data_check,
    )

    lozo_sha = _resolve_lozo_sha(lozo_root)
    if args.expected_lozo_ref and lozo_sha != args.expected_lozo_ref:
        raise RuntimeError(
            "LOZO ref mismatch: "
            f"expected {args.expected_lozo_ref}, got {lozo_sha or 'unknown'}"
        )

    manifest = {
        "created_at": time.time(),
        "profile": args.profile,
        "lozo_root": str(lozo_root),
        "lozo_sha": lozo_sha,
        "expected_lozo_ref": args.expected_lozo_ref,
        "medium_models_dir": str(medium_models_dir),
        "run_root": str(run_root),
        "args": vars(args),
        "num_runs": len(run_specs),
        "runs": [asdict(spec) for spec in run_specs],
    }
    _write_json(run_root / "manifest.json", manifest)

    counts = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "no_marker": 0,
        "dry_run": 0,
    }
    run_records: list[dict[str, Any]] = []
    for idx, spec in enumerate(run_specs, start=1):
        print(
            f"[{idx:03d}/{len(run_specs):03d}] {spec.method} "
            f"task={spec.task} k={spec.k} seed={spec.seed}"
        )
        print(f"  data_dir={spec.data_dir}")
        print(f"  log_path={spec.log_path}")
        if _is_completed(spec) and not args.force:
            marker = Path(spec.result_dir) / spec.completion_glob
            print(f"  skip: completion marker present at {marker}")
            counts["skipped"] += 1
            run_records.append(
                {
                    "status": "skipped",
                    "task": spec.task,
                    "k": spec.k,
                    "seed": spec.seed,
                    "method": spec.method,
                    "data_dir": spec.data_dir,
                    "result_dir": spec.result_dir,
                    "log_path": spec.log_path,
                    "metrics": _extract_metrics(spec),
                }
            )
            continue

        status, payload = _run_one(
            spec=spec,
            medium_models_dir=medium_models_dir,
            dry_run=args.dry_run,
            poll_gpu_memory_sec=args.poll_gpu_memory_sec,
            log_tail_lines=args.log_tail_lines,
        )
        counts[status] = counts.get(status, 0) + 1
        run_records.append(payload)
        rc = payload.get("return_code", "N/A")
        elapsed = payload.get("elapsed_sec", 0)
        print(f"  status={status} rc={rc} elapsed={elapsed:.1f}s")
        for hint in payload.get("failure_hints", []):
            print(f"  hint: {hint}")
        if status in {"failed", "no_marker"}:
            tail = payload.get("log_tail", [])
            if tail:
                print("  log_tail:")
                for line in tail[-12:]:
                    print(f"    {line}")

    summary = {
        "completed_at": time.time(),
        "lozo_sha": lozo_sha,
        "counts": counts,
        "num_runs": len(run_specs),
        "records": run_records,
    }
    _write_json(run_root / "summary.json", summary)
    print("\nRun complete.")
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {run_root / 'manifest.json'}")
    print(f"Summary : {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()
