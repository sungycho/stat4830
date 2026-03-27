"""Orchestrate LOZO medium-model MeZO/LOZO experiment suites.

This runner is intentionally script-centric (week8 style): it builds a run
matrix, executes upstream `medium_models/mezo.sh` and `medium_models/lozo.sh`,
and records reproducibility artifacts in a local run root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ["SST-2", "sst-5", "SNLI", "MNLI", "RTE", "trec"]
DEFAULT_SEEDS = [13, 21, 42, 87, 100]
DEFAULT_K_VALUES = [16, 512]


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
    # upstream writes files like test_results_<task>.txt
    return f"test_results_{task}.txt"


def _expected_paths(
    medium_models_dir: Path,
    task: str,
    k: int,
    seed: int,
    model_name: str,
    cfg: MethodConfig,
) -> tuple[Path, str, str, Path]:
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
    log_dir = medium_models_dir / "log_dir"
    log_path = log_dir / f"{task}-{gr_tag}-{tag}.log"
    return result_dir, gr_tag, tag, log_path


def _build_run_spec(
    medium_models_dir: Path,
    task: str,
    k: int,
    seed: int,
    model: str,
    model_name: str,
    cfg: MethodConfig,
) -> RunSpec:
    result_dir, _, _, log_path = _expected_paths(
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
        result_dir=str(result_dir),
        completion_glob=_completion_glob(task),
        log_path=str(log_path),
    )


def _is_completed(spec: RunSpec) -> bool:
    result_dir = Path(spec.result_dir)
    marker = result_dir / spec.completion_glob
    return marker.exists()


def _extract_metrics(spec: RunSpec) -> dict[str, float]:
    result_dir = Path(spec.result_dir)
    marker = result_dir / spec.completion_glob
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
                    spec = _build_run_spec(
                        medium_models_dir=medium_models_dir,
                        task=task,
                        k=k,
                        seed=seed,
                        model=args.model,
                        model_name=args.model_name,
                        cfg=method_configs[method],
                    )
                    run_specs.append(spec)

    if args.max_runs is not None:
        run_specs = run_specs[: args.max_runs]
    return run_specs


def _run_one(
    spec: RunSpec,
    medium_models_dir: Path,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    env = os.environ.copy()
    env.update(spec.env)

    started_at = time.time()
    payload: dict[str, Any] = {
        "task": spec.task,
        "k": spec.k,
        "seed": spec.seed,
        "method": spec.method,
        "script": spec.script,
        "command": spec.command,
        "env": spec.env,
        "result_dir": spec.result_dir,
        "log_path": spec.log_path,
        "started_at": started_at,
    }

    if dry_run:
        payload["status"] = "dry_run"
        payload["finished_at"] = time.time()
        payload["elapsed_sec"] = payload["finished_at"] - started_at
        return "dry_run", payload

    proc = subprocess.run(
        spec.command,
        cwd=medium_models_dir,
        env=env,
        check=False,
    )
    payload["return_code"] = proc.returncode
    payload["finished_at"] = time.time()
    payload["elapsed_sec"] = payload["finished_at"] - started_at
    if proc.returncode == 0 and _is_completed(spec):
        payload["status"] = "completed"
        payload["metrics"] = _extract_metrics(spec)
        return "completed", payload
    if proc.returncode == 0 and not _is_completed(spec):
        payload["status"] = "no_marker"
        return "no_marker", payload
    payload["status"] = "failed"
    return "failed", payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LOZO medium-model suite in a reproducible, resumable way.",
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

    # Upstream medium_models defaults from README examples.
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

    # Long/full and short/smoke schedules.
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lozo_root = Path(args.lozo_root).expanduser().resolve()
    medium_models_dir = lozo_root / "medium_models"
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    if not medium_models_dir.exists():
        raise FileNotFoundError(
            f"Could not find medium_models at {medium_models_dir}. "
            "Clone LOZO first (e.g. git clone "
            "https://github.com/optsuite/LOZO external/LOZO)."
        )

    run_specs = build_run_matrix(args, medium_models_dir)
    manifest = {
        "created_at": time.time(),
        "profile": args.profile,
        "lozo_root": str(lozo_root),
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
        )
        counts[status] = counts.get(status, 0) + 1
        run_records.append(payload)
        rc = payload.get("return_code", "N/A")
        elapsed = payload.get("elapsed_sec", 0)
        print(f"  status={status} rc={rc} elapsed={elapsed:.1f}s")

    summary = {
        "completed_at": time.time(),
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
