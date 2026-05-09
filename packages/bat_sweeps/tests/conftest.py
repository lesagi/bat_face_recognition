"""Shared fixtures + sys.path wiring for bat_sweeps tests.

Each test gates ``optuna``/``torch`` via ``pytest.importorskip`` so
collection stays graceful on minimal environments. The package roots
are added to ``sys.path`` so tests work without ``uv sync``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
for pkg in (
    "bat_core",
    "bat_data",
    "bat_evaluation",
    "bat_losses",
    "bat_models",
    "bat_training",
    "bat_sweeps",
    "bat_tracking",
):
    pkg_root = _REPO_ROOT / "packages" / pkg
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "true")
