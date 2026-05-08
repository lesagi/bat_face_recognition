"""CosFace loss (Wang et al., CVPR 2018).

Large margin cosine ``m`` is subtracted from the target-class cosine inside
:class:`bat_models.CosFaceHead`; this class is cross-entropy over the logits.

Reference defaults (head-side): ``s = 64.0``, ``m = 0.35``.
"""

from __future__ import annotations

from bat_losses._ce_base import _MarginCrossEntropy


class CosFaceLoss(_MarginCrossEntropy):
    """Cross-entropy over CosFace head logits."""
