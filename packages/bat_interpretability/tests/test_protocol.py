"""Verify all adapters satisfy ``bat_core.InterpretabilityAdapter``."""

from __future__ import annotations

import pytest

# These tests don't need torch at runtime — only the protocol check.
from bat_core.interfaces import InterpretabilityAdapter
from bat_interpretability import EmbeddingProjectionAdapter, GradCAMAdapter, SiameseSaliencyAdapter


def test_siamese_saliency_satisfies_protocol() -> None:
    adapter = SiameseSaliencyAdapter()
    assert isinstance(adapter, InterpretabilityAdapter)


def test_gradcam_satisfies_protocol() -> None:
    adapter = GradCAMAdapter()
    assert isinstance(adapter, InterpretabilityAdapter)


def test_embedding_projection_satisfies_protocol() -> None:
    adapter = EmbeddingProjectionAdapter()
    assert isinstance(adapter, InterpretabilityAdapter)


@pytest.mark.parametrize(
    "method",
    ["vanilla", "guided", "integrated_gradients", "smoothgrad"],
)
def test_siamese_saliency_method_options(method: str) -> None:
    """Each of the four documented methods constructs cleanly."""
    adapter = SiameseSaliencyAdapter(method=method)  # type: ignore[arg-type]
    assert adapter.method == method
