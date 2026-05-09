"""Regression test: every plot helper routes filenames through bat_stats.naming.

If this test fails, an artifact has escaped with a generic name -- precisely
the bug Worker G was chartered to fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

from bat_reporting import plots  # noqa: E402

EXPECTED_TOKENS = ("rousettus", "video", "random", "arcface", "arcface")


def _assert_signature(filename: str) -> None:
    for token in EXPECTED_TOKENS:
        assert token in filename, f"missing token {token!r} in {filename}"


def test_training_curves_filename(tmp_path: Path, cfg: dict[str, Any]) -> None:
    history = {"train": {"loss": [1.0, 0.5, 0.3]}, "val": {"loss": [1.1, 0.6, 0.4]}}
    out = plots.plot_training_curves(history, cfg, tmp_path)
    assert out.exists() and out.stat().st_size > 0
    _assert_signature(out.name)


def test_confusion_filename(tmp_path: Path, cfg: dict[str, Any]) -> None:
    out = plots.plot_confusion(10, 2, 14, 3, threshold=0.5, cfg=cfg, output_dir=tmp_path)
    assert out.exists()
    _assert_signature(out.name)


def test_roc_filename(tmp_path: Path, cfg: dict[str, Any]) -> None:
    out = plots.plot_roc([0.0, 0.1, 1.0], [0.0, 0.7, 1.0], 0.85, cfg, tmp_path)
    assert out.exists()
    _assert_signature(out.name)


def test_cmc_filename(tmp_path: Path, cfg: dict[str, Any]) -> None:
    out = plots.plot_cmc([0.6, 0.7, 0.8, 0.9], cfg, tmp_path)
    assert out.exists()
    _assert_signature(out.name)


def test_permutation_null_filename(tmp_path: Path, cfg: dict[str, Any]) -> None:
    out = plots.plot_permutation_null("f1", 0.8, [0.4, 0.45, 0.5, 0.55], cfg, tmp_path)
    assert out.exists()
    _assert_signature(out.name)
    assert out.name.startswith(
        "null_dist_f1__"
    ), "stem must be slugified before the experiment signature"
