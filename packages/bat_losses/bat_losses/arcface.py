"""ArcFace loss (Deng et al., CVPR 2019).

The margin (additive angular margin ``m``) and scale ``s`` are applied by the
:class:`bat_models.ArcFaceHead`; this class is cross-entropy over those
already-margined logits, exposed here for symmetry with other embedding-family
losses and to declare ``family = "embedding"``.

Reference defaults (head-side): ``s = 64.0``, ``m = 0.50``.
"""

from __future__ import annotations

from bat_losses._ce_base import _MarginCrossEntropy


class ArcFaceLoss(_MarginCrossEntropy):
    """Cross-entropy over ArcFace head logits."""
