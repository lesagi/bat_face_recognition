"""bat_interpretability — adapters that satisfy ``bat_core.InterpretabilityAdapter``.

This package ports the TF/Keras saliency logic from ``app/visualization/`` to
PyTorch (``torch.autograd.grad``) and adds Grad-CAM and embedding-projection
adapters used by the unified post-training PDF report.

Public surface:

* :class:`SiameseSaliencyAdapter` — pair-family gradient saliency
  (``vanilla`` / ``guided`` / ``integrated_gradients`` / ``smoothgrad``).
* :class:`GradCAMAdapter` — embedding-family Grad-CAM on the last
  convolutional layer of the backbone.
* :class:`EmbeddingProjectionAdapter` — t-SNE + UMAP 2-D projection.
* :func:`make_composite` — 5-identities-per-page layout (original | saliency).
* :func:`select_adapter` — picks the right adapter from ``model.family``.
"""

from __future__ import annotations

from bat_interpretability.composite import make_composite
from bat_interpretability.dispatcher import select_adapter
from bat_interpretability.gradcam import GradCAMAdapter
from bat_interpretability.projection import EmbeddingProjectionAdapter
from bat_interpretability.siamese_saliency import SiameseSaliencyAdapter

__all__ = [
    "EmbeddingProjectionAdapter",
    "GradCAMAdapter",
    "SiameseSaliencyAdapter",
    "make_composite",
    "select_adapter",
]
