#!/usr/bin/env python3
"""Static check: PyTorch packages must not depend on legacy TF code.

Run before Phase-4 step 4 (deleting ``app/``, ``run.py``, ``Makefile``,
``scripts/``, ``setup.py``) to confirm nothing in ``packages/`` or
``configs/`` references the legacy TF surfaces. Exits non-zero on any
violation, printing every offending file:line.

Usage:
    python tools/check_tf_isolation.py
    python tools/check_tf_isolation.py --root /path/to/repo

Designed to run with stdlib only --- no PyTorch / no uv sync required.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

LEGACY_ROOTS = ("app", "legacy")
PYTHON_IMPORT_PATTERNS = tuple(
    re.compile(rf"^\s*(?:from|import)\s+{root}(?:\.|\s|$)") for root in LEGACY_ROOTS
)
PATH_REFERENCE_PATTERNS = tuple(
    re.compile(rf"(?<![A-Za-z0-9_/]){root}/") for root in LEGACY_ROOTS
)
SCANNED_PACKAGE_DIR = "packages"
SCANNED_CONFIG_DIR = "configs"


def _iter_files(root: Path, subdir: str, suffixes: tuple[str, ...]) -> Iterable[Path]:
    base = root / subdir
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if path.suffix in suffixes and path.is_file():
            yield path


def _scan_python(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern in PYTHON_IMPORT_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, line.strip()))
                break
    return hits


def _scan_yaml(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern in PATH_REFERENCE_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, line.strip()))
                break
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repo root (default: cwd)."
    )
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    violations: list[str] = []

    for path in _iter_files(root, SCANNED_PACKAGE_DIR, (".py",)):
        for lineno, line in _scan_python(path):
            violations.append(f"{path.relative_to(root)}:{lineno}: {line}")

    for path in _iter_files(root, SCANNED_CONFIG_DIR, (".yaml", ".yml")):
        for lineno, line in _scan_yaml(path):
            violations.append(f"{path.relative_to(root)}:{lineno}: {line}")

    if violations:
        print(f"Found {len(violations)} legacy reference(s):", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print(
        f"OK: scanned {SCANNED_PACKAGE_DIR}/ + {SCANNED_CONFIG_DIR}/; "
        f"no imports or path refs to {' / '.join(LEGACY_ROOTS)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
