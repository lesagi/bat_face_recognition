"""Shared fixtures + dependency guards for the bat_data test suite.

The tests are designed to run with only ``pydantic`` + ``pytest`` (and
the package's own pure-Python deps) available. Anything that needs
torch / cv2 / pandas is gated by ``pytest.importorskip``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make every package importable when running pytest from the repo root
# without an editable install (the harness has no `uv sync`).
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
for pkg in ("bat_core", "bat_data"):
    pkg_root = _REPO_ROOT / "packages" / pkg
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

# Older Python on CI may complain about pydantic plugin discovery; keep
# the env clean.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "true")
