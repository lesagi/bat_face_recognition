"""Sub-center ArcFace loss (Deng et al., ECCV 2020).

The K-sub-center max-pool over per-class prototypes is performed by
:class:`bat_models.SubCenterArcFaceHead`; this class is cross-entropy over
the resulting margined logits.

Reference defaults (head-side): ``s = 64.0``, ``m = 0.50``, ``K = 3``.
"""

from __future__ import annotations

from bat_losses._ce_base import _MarginCrossEntropy


class SubCenterArcFaceLoss(_MarginCrossEntropy):
    """Cross-entropy over sub-center ArcFace head logits."""
