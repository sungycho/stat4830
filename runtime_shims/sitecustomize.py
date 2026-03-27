"""Runtime compatibility shim for upstream LOZO imports.

Python auto-imports ``sitecustomize`` when this directory is on PYTHONPATH.
This keeps compatibility logic isolated from project packages.
"""

from __future__ import annotations

try:
    from transformers import file_utils as _file_utils  # type: ignore
except Exception:  # pragma: no cover
    _file_utils = None

if _file_utils is not None and not hasattr(
    _file_utils,
    "is_torch_tpu_available",
):
    _compat_symbol = None
    try:
        from transformers.utils import import_utils as _import_utils  # type: ignore

        _compat_symbol = getattr(
            _import_utils,
            "is_torch_tpu_available",
            None,
        )
    except Exception:  # pragma: no cover
        _compat_symbol = None

    if _compat_symbol is None:

        def _compat_symbol() -> bool:
            return False

    setattr(
        _file_utils,
        "is_torch_tpu_available",
        _compat_symbol,
    )
