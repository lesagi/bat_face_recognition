"""AdaFace loss (Kim et al., CVPR 2022).

Adaptive margin is applied by :class:`bat_models.AdaFaceHead` (margin scaled
by feature-norm statistics); this class is cross-entropy over those logits.

Reference defaults (head-side): ``s = 64.0``, ``m = 0.40``, ``h = 0.333``.
"""

from __future__ import annotations

from bat_losses._ce_base import _MarginCrossEntropy


class AdaFaceLoss(_MarginCrossEntropy):
    """Cross-entropy over AdaFace head logits."""
