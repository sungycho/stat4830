"""Force `import src.*` to resolve against LOZO medium_models sources.

This avoids conflicts with this repository's own `src` package when running
upstream LOZO scripts in the same Python environment.
"""

from __future__ import annotations

import os
from pathlib import Path

_base = os.environ.get("LOZO_MEDIUM_MODELS_DIR", "").strip()
if _base:
    __path__ = [str(Path(_base).expanduser().resolve() / "src")]
else:  # pragma: no cover
    __path__ = []
