"""Dispatch from ``model.family`` to the right interpretability adapter.

The plan calls for **one adapter per family**:

- ``family == "pair"``      → :class:`SiameseSaliencyAdapter`
- ``family == "embedding"`` → :class:`GradCAMAdapter`

:class:`EmbeddingProjectionAdapter` is *family-independent* — it always works
on any embedding — and is therefore returned by :func:`available_adapters`
alongside the family-specific one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bat_interpretability.gradcam import GradCAMAdapter
from bat_interpretability.projection import EmbeddingProjectionAdapter
from bat_interpretability.siamese_saliency import SiameseSaliencyAdapter

if TYPE_CHECKING:  # pragma: no cover
    from bat_core.interfaces import FaceModel, InterpretabilityAdapter


def select_adapter(model: FaceModel) -> InterpretabilityAdapter:
    """Return the family-specific adapter for ``model``.

    Raises :class:`ValueError` if ``model.family`` is unknown.
    """
    family = getattr(model, "family", None)
    if family == "pair":
        return SiameseSaliencyAdapter()
    if family == "embedding":
        return GradCAMAdapter()
    raise ValueError(
        f"Unknown model.family={family!r}; expected 'pair' or 'embedding'."
    )


def available_adapters(model: FaceModel) -> list[InterpretabilityAdapter]:
    """Return all adapters applicable to ``model``.

    Always includes :class:`EmbeddingProjectionAdapter` (family-independent)
    plus the family-specific saliency adapter.
    """
    return [select_adapter(model), EmbeddingProjectionAdapter()]
