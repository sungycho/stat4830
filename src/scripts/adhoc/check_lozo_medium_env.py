"""Preflight checks for LOZO medium-model runs.

This script validates local binaries, Python dependencies, upstream LOZO files,
and (optionally) k-shot dataset directories expected by the suite runner.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT_BINPATH = str(REPO_ROOT / "runtime_shims" / "bin")


def _parse_csv_text(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _check_binaries(names: list[str]) -> list[str]:
    search_path = (
        f"{COMPAT_BINPATH}:{os.environ.get('PATH', '')}"
        if os.environ.get("PATH", "")
        else COMPAT_BINPATH
    )
    missing: list[str] = []
    for name in names:
        if shutil.which(name, path=search_path) is None:
            missing.append(name)
    return missing


def _check_python_modules(names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _check_required_files(medium_models_dir: Path) -> list[Path]:
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
    return [p for p in required if not p.exists()]


def _check_dataset_dirs(
    medium_models_dir: Path,
    tasks: list[str],
    k_values: list[int],
    seeds: list[int],
) -> list[Path]:
    root = medium_models_dir / "data" / "k-shot-1k-test"
    missing: list[Path] = []
    for task in tasks:
        for k in k_values:
            for seed in seeds:
                path = root / task / f"{k}-{seed}"
                if not path.exists():
                    missing.append(path)
    return missing


def _lozo_sha(lozo_root: Path) -> str | None:
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


def _parse_major(version: str) -> int | None:
    head = version.strip().split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _check_transformers_lozo_compat() -> tuple[bool, str]:
    try:
        from transformers import file_utils as _file_utils  # type: ignore
        from transformers.pytorch_utils import (  # type: ignore
            find_pruneable_heads_and_indices,
        )
        from transformers.utils import import_utils as _import_utils  # type: ignore
    except Exception as exc:
        return (False, f"{type(exc).__name__}: {exc}")
    if (
        not hasattr(_file_utils, "is_torch_tpu_available")
        and not hasattr(_import_utils, "is_torch_tpu_available")
    ):
        return (
            True,
            "is_torch_tpu_available missing in transformers modules "
            "(runner compatibility shim will inject fallback)",
        )
    _ = find_pruneable_heads_and_indices
    return (True, "")


def _check_lozo_runtime_shim(repo_root: Path) -> tuple[bool, str]:
    """Verify ``runtime_shims/sitecustomize`` backports LOZO import expectations."""
    shim = str(repo_root / "runtime_shims")
    code = (
        "from transformers.optimization import AdamW; "
        "from transformers.file_utils import is_torch_tpu_available; "
        "from transformers.pytorch_utils import find_pruneable_heads_and_indices; "
        "assert callable(AdamW); assert callable(is_torch_tpu_available)"
    )
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{shim}:{prior}" if prior else shim
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover
        return (False, f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return (
            False,
            err or "subprocess failed with no stderr/stdout",
        )
    return (True, "")


def _check_lozo_src_resolution(
    repo_root: Path,
    medium_models_dir: Path,
) -> tuple[bool, str]:
    """Verify `import src.*` resolves to LOZO medium_models sources."""
    compat = str(repo_root / "runtime_shims")
    lozo_src_shim = str(repo_root / "runtime_shims" / "lozo_src")
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    prefix = f"{compat}:{lozo_src_shim}"
    env["PYTHONPATH"] = f"{prefix}:{prior}" if prior else prefix
    env["LOZO_MEDIUM_MODELS_DIR"] = str(medium_models_dir)
    code = (
        "import src.modeling_roberta as m; "
        "print(getattr(m, '__file__', ''))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(medium_models_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover
        return (False, f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return (
            False,
            err or "subprocess failed with no stderr/stdout",
        )
    return (True, "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check LOZO medium-model prerequisites before running.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lozo-root", default="external/LOZO")
    p.add_argument("--tasks", default="SST-2,RTE")
    p.add_argument("--k-values", default="16")
    p.add_argument("--seeds", default="42")
    p.add_argument(
        "--require-data",
        action="store_true",
        help="Fail if any expected k-shot dataset directory is missing.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing optional checks (e.g., nvidia-smi).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lozo_root = Path(args.lozo_root).expanduser().resolve()
    medium_models_dir = lozo_root / "medium_models"
    tasks = _parse_csv_text(args.tasks)
    k_values = _parse_csv_ints(args.k_values)
    seeds = _parse_csv_ints(args.seeds)

    print("[check] LOZO root:", lozo_root)
    print("[check] medium_models:", medium_models_dir)

    errors: list[str] = []
    warnings: list[str] = []

    if not medium_models_dir.exists():
        errors.append(
            f"Missing medium_models directory: {medium_models_dir}. "
            "Clone LOZO or fix --lozo-root."
        )
    else:
        missing_files = _check_required_files(medium_models_dir)
        if missing_files:
            errors.append(
                "Missing required LOZO files:\n"
                + "\n".join(f"  - {p}" for p in missing_files[:20])
            )

    missing_bins = _check_binaries(["bash", "git", "jq"])
    if missing_bins:
        errors.append(
            "Missing required binaries: "
            + ", ".join(missing_bins)
            + ". Install with apt-get or your OS package manager."
        )

    missing_mods = _check_python_modules(
        [
            "torch",
            "transformers",
            "datasets",
            "filelock",
            "loralib",
            "sklearn",
            "pandas",
            "loguru",
        ]
    )
    if missing_mods:
        errors.append(
            "Missing required Python modules: "
            + ", ".join(missing_mods)
            + ". Run `uv sync` then retry."
        )
    else:
        try:
            transformers_mod = importlib.import_module("transformers")
            transformers_version = str(
                getattr(transformers_mod, "__version__", "")
            )
            major = _parse_major(transformers_version)
            if major is not None and major >= 5:
                errors.append(
                    "Incompatible transformers version detected: "
                    f"{transformers_version}. "
                    "LOZO medium code expects transformers<5. "
                    "Run `uv sync` after pulling latest repo changes."
                )
            ok, reason = _check_transformers_lozo_compat()
            if not ok:
                errors.append(
                    "Transformers LOZO compatibility check failed "
                    f"(installed: {transformers_version}): {reason}. "
                    "Run `uv sync` after pulling latest repo changes."
                )
            elif reason:
                warnings.append(reason)
            ok_shim, shim_err = _check_lozo_runtime_shim(REPO_ROOT)
            if not ok_shim:
                errors.append(
                    "LOZO runtime compatibility shim check failed "
                    f"(runtime_shims on PYTHONPATH): {shim_err}. "
                    "Ensure repo contains runtime_shims/sitecustomize.py and "
                    "run training via src.scripts.run_lozo_medium_suite "
                    "(which prepends that path)."
                )
            elif medium_models_dir.exists():
                ok_src, src_err = _check_lozo_src_resolution(
                    REPO_ROOT,
                    medium_models_dir,
                )
                if not ok_src:
                    errors.append(
                        "LOZO source import check failed "
                        "(src.modeling_roberta unresolved): "
                        f"{src_err}. "
                        "This can happen when another installed `src` package "
                        "shadows external/LOZO/medium_models/src."
                    )
        except Exception:
            pass

    if shutil.which("nvidia-smi") is None:
        msg = "nvidia-smi not found. GPU memory plots may be unavailable."
        if args.strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if args.require_data and medium_models_dir.exists():
        missing_data = _check_dataset_dirs(
            medium_models_dir=medium_models_dir,
            tasks=tasks,
            k_values=k_values,
            seeds=seeds,
        )
        if missing_data:
            errors.append(
                "Missing expected k-shot dataset directories. "
                "Run LOZO data prep first:\n"
                "  cd external/LOZO/medium_models && "
                "python tools/generate_k_shot_data.py "
                "--mode k-shot-1k-test --k 16\n"
                "Examples missing:\n"
                + "\n".join(f"  - {p}" for p in missing_data[:10])
            )

    sha = _lozo_sha(lozo_root)
    if sha:
        print("[check] LOZO git SHA:", sha)
    else:
        warnings.append("Could not resolve LOZO git SHA (non-git checkout?).")

    if warnings:
        print("\n[warn]")
        for msg in warnings:
            print("-", msg)

    if errors:
        print("\n[error]")
        for msg in errors:
            print("-", msg)
        return 1

    print("\n[ok] LOZO preflight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
