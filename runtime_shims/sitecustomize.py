"""Runtime compatibility shim for upstream LOZO imports.

Python auto-imports ``sitecustomize`` when this directory is on PYTHONPATH.
This keeps compatibility logic isolated from project packages.

LOZO upstream was tested with ``transformers==4.28.1`` (Python ~3.9).  That
stack does not install cleanly on Python 3.12+; this project instead pins
``transformers>=4.30,<5`` and backports a few removed symbols here so
upstream LOZO files stay unmodified.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG_PATH = _REPO_ROOT / ".cursor" / "debug-a13946.log"
DEBUG_SESSION_ID = "a13946"


def _debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
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


_run_id = f"sitecustomize:{int(time.time())}"

# region agent log
_debug_log(
    run_id=_run_id,
    hypothesis_id="H2",
    location="runtime_shims/sitecustomize.py:module",
    message="sitecustomize loaded",
    data={},
)
# endregion

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
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H2",
        location="runtime_shims/sitecustomize.py:tpu_patch",
        message=(
            "Patched transformers.file_utils."
            "is_torch_tpu_available"
        ),
        data={},
    )
    # endregion

try:
    import transformers as _transformers  # type: ignore

    if not hasattr(_transformers, "is_torch_tpu_available"):
        _top_level_symbol = None
        if _file_utils is not None and hasattr(
            _file_utils,
            "is_torch_tpu_available",
        ):
            _top_level_symbol = getattr(
                _file_utils,
                "is_torch_tpu_available",
            )
        if _top_level_symbol is None:

            def _top_level_symbol() -> bool:
                return False

        setattr(
            _transformers,
            "is_torch_tpu_available",
            _top_level_symbol,
        )
        if hasattr(_transformers, "_objects"):
            try:
                _transformers._objects[  # type: ignore[attr-defined]
                    "is_torch_tpu_available"
                ] = _top_level_symbol
            except Exception:
                pass
        # region agent log
        _debug_log(
            run_id=_run_id,
            hypothesis_id="H8",
            location="runtime_shims/sitecustomize.py:top_level_tpu_patch",
            message="Patched transformers.is_torch_tpu_available",
            data={},
        )
        # endregion
except Exception as _exc:  # pragma: no cover
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H8",
        location="runtime_shims/sitecustomize.py:top_level_tpu_patch_error",
        message="Failed to patch transformers.is_torch_tpu_available",
        data={"error": repr(_exc)},
    )
    # endregion

try:
    from transformers.utils import import_utils as _tf_import_utils  # type: ignore

    _orig_lazy_getattr = _tf_import_utils._LazyModule.__getattr__

    def _compat_lazy_getattr(self, name):
        if name == "is_torch_tpu_available":
            try:
                from transformers import file_utils as _fu  # type: ignore

                return getattr(_fu, "is_torch_tpu_available")
            except Exception:

                def _fallback() -> bool:
                    return False

                return _fallback
        return _orig_lazy_getattr(self, name)

    _tf_import_utils._LazyModule.__getattr__ = _compat_lazy_getattr
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H9",
        location="runtime_shims/sitecustomize.py:lazy_getattr_patch",
        message="Patched transformers lazy getattr for TPU symbol",
        data={},
    )
    # endregion
except Exception as _exc:  # pragma: no cover
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H9",
        location="runtime_shims/sitecustomize.py:lazy_getattr_patch_error",
        message="Failed to patch transformers lazy getattr",
        data={"error": repr(_exc)},
    )
    # endregion

try:
    from transformers import optimization as _optimization  # type: ignore
    from torch.optim import AdamW as _torch_adamw  # type: ignore

    if not hasattr(_optimization, "AdamW"):
        setattr(_optimization, "AdamW", _torch_adamw)
        # region agent log
        _debug_log(
            run_id=_run_id,
            hypothesis_id="H1",
            location="runtime_shims/sitecustomize.py:adamw_patch",
            message="Patched transformers.optimization.AdamW",
            data={},
        )
        # endregion
except Exception as _exc:  # pragma: no cover
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H1",
        location="runtime_shims/sitecustomize.py:adamw_patch_error",
        message="Failed to patch transformers.optimization.AdamW",
        data={"error": repr(_exc)},
    )
    # endregion

try:
    from transformers.training_args import (  # type: ignore
        TrainingArguments as _TrainingArguments,
    )

    _orig_post_init = _TrainingArguments.__post_init__

    def _compat_post_init(self):
        _orig_post_init(self)
        if getattr(self, "local_rank", None) == 0:
            local_rank_env = os.environ.get("LOCAL_RANK")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if local_rank_env is None and world_size <= 1:
                self.local_rank = -1

    _TrainingArguments.__post_init__ = _compat_post_init
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H7",
        location="runtime_shims/sitecustomize.py:training_args_patch",
        message="Patched TrainingArguments.__post_init__ for local_rank",
        data={},
    )
    # endregion
except Exception as _exc:  # pragma: no cover
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H7",
        location="runtime_shims/sitecustomize.py:training_args_patch_error",
        message="Failed to patch TrainingArguments.__post_init__",
        data={"error": repr(_exc)},
    )
    # endregion

try:
    import torch as _torch  # type: ignore

    _orig_torch_load = _torch.load

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _torch.load = _compat_torch_load
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H6",
        location="runtime_shims/sitecustomize.py:torch_load_patch",
        message="Patched torch.load default weights_only=False",
        data={},
    )
    # endregion
except Exception as _exc:  # pragma: no cover
    # region agent log
    _debug_log(
        run_id=_run_id,
        hypothesis_id="H6",
        location="runtime_shims/sitecustomize.py:torch_load_patch_error",
        message="Failed to patch torch.load",
        data={"error": repr(_exc)},
    )
    # endregion
