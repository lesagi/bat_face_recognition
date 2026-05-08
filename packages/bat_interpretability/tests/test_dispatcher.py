"""Test ``select_adapter`` and ``available_adapters`` dispatch logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from bat_interpretability import (
    EmbeddingProjectionAdapter,
    GradCAMAdapter,
    SiameseSaliencyAdapter,
    select_adapter,
)
from bat_interpretability.dispatcher import available_adapters


@dataclass
class _MockModel:
    """Mock that satisfies ``bat_core.FaceModel`` only structurally."""

    family: Literal["pair", "embedding"]

    def forward_embedding(self, x):  # pragma: no cover - never called here
        return x

    def forward_train(self, x, labels):  # pragma: no cover
        return x

    def export_for_inference(self):  # pragma: no cover
        return self


def test_dispatch_pair_returns_siamese_saliency() -> None:
    adapter = select_adapter(_MockModel(family="pair"))
    assert isinstance(adapter, SiameseSaliencyAdapter)


def test_dispatch_embedding_returns_gradcam() -> None:
    adapter = select_adapter(_MockModel(family="embedding"))
    assert isinstance(adapter, GradCAMAdapter)


def test_dispatch_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model.family"):
        select_adapter(_MockModel(family="weird"))  # type: ignore[arg-type]


def test_available_adapters_includes_projection() -> None:
    adapters = available_adapters(_MockModel(family="embedding"))
    assert any(isinstance(a, EmbeddingProjectionAdapter) for a in adapters)
    assert any(isinstance(a, GradCAMAdapter) for a in adapters)
