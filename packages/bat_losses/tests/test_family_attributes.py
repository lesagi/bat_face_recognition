"""Each loss declares the right ``.family`` attribute (matches plan)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from bat_losses import (  # noqa: E402
    AdaFaceLoss,
    ArcFaceLoss,
    BCELoss,
    CosFaceLoss,
    FocalLoss,
    SubCenterArcFaceLoss,
    TripletLoss,
)

PAIR_LOSSES = [BCELoss, FocalLoss, TripletLoss]
EMBEDDING_LOSSES = [ArcFaceLoss, AdaFaceLoss, CosFaceLoss, SubCenterArcFaceLoss]


def test_pair_losses_declare_pair_family() -> None:
    for cls in PAIR_LOSSES:
        assert cls.family == "pair", f"{cls.__name__} should be family='pair'"
        assert cls().family == "pair"


def test_embedding_losses_declare_embedding_family() -> None:
    for cls in EMBEDDING_LOSSES:
        assert cls.family == "embedding", (
            f"{cls.__name__} should be family='embedding'"
        )
        assert cls().family == "embedding"
