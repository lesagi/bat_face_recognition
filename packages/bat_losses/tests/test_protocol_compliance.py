"""Each loss class satisfies bat_core.Loss runtime-checkable protocol."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from bat_core import Loss  # noqa: E402
from bat_losses import (  # noqa: E402
    AdaFaceLoss,
    ArcFaceLoss,
    BCELoss,
    CosFaceLoss,
    FocalLoss,
    SubCenterArcFaceLoss,
    TripletLoss,
)

ALL_LOSSES = [
    BCELoss,
    FocalLoss,
    TripletLoss,
    ArcFaceLoss,
    AdaFaceLoss,
    CosFaceLoss,
    SubCenterArcFaceLoss,
]


def test_every_loss_satisfies_loss_protocol() -> None:
    for cls in ALL_LOSSES:
        assert isinstance(cls(), Loss), f"{cls.__name__} fails Loss protocol"
