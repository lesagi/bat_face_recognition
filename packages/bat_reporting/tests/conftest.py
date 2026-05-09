"""Shared fixtures for bat_reporting tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bat_core.types import (
    EvalReport,
    IdentificationMetrics,
    Predictions,
    SaliencyImage,
    VerificationMetrics,
)


def _build_cfg() -> dict[str, Any]:
    return {
        "data": {"species": "rousettus", "source": "video", "background": "random"},
        "model": {"name": "arcface"},
        "loss": {"name": "arcface"},
        "trainer": {"epochs": 4, "lr": 1e-3},
    }


@pytest.fixture
def cfg() -> dict[str, Any]:
    return _build_cfg()


def _synth_predictions(n: int = 40, seed: int = 0) -> Predictions:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(np.int64)
    # Push positives toward higher scores so AUC > chance.
    y_score = rng.uniform(0, 1, size=n) + 0.4 * y_true
    y_score = np.clip(y_score, 0.0, 1.0)
    return Predictions(
        y_true=tuple(int(v) for v in y_true),
        y_score=tuple(float(v) for v in y_score),
    )


@pytest.fixture
def eval_report() -> EvalReport:
    preds = _synth_predictions()
    verif = VerificationMetrics(
        roc_auc=0.83,
        youden_j=0.5,
        optimal_threshold=0.55,
        tar_at_far_1e3=0.42,
        tar_at_far_1e4=0.21,
    )
    ident = IdentificationMetrics(
        top1=0.71,
        top5=0.92,
        map=0.66,
        cmc=tuple(min(1.0, 0.71 + 0.03 * k) for k in range(10)),
    )
    return EvalReport(verification=verif, identification=ident, predictions=preds)


@pytest.fixture
def saliency_images(tmp_path: Path) -> list[SaliencyImage]:
    """Build 5 SaliencyImage entries backed by synthetic 32x32 PNGs."""

    from PIL import Image

    items: list[SaliencyImage] = []
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    for idx in range(5):
        img_arr = (rng.uniform(0, 255, size=(32, 32, 3))).astype(np.uint8)
        path = img_dir / f"identity_{idx}.png"
        Image.fromarray(img_arr).save(str(path))
        sal = rng.uniform(0, 1, size=(32, 32)).astype(np.float32)
        items.append(
            SaliencyImage(
                identity=f"id_{idx}",
                image_path=path,
                saliency=sal,
                method="vanilla",
            )
        )
    return items
