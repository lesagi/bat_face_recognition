"""bat_losses — pair and embedding family losses for bat face recognition.

Pair-family (work on similarity / pair predictions):
  - :class:`BCELoss`
  - :class:`FocalLoss`
  - :class:`TripletLoss`

Embedding-family (cross-entropy over margin-head logits):
  - :class:`ArcFaceLoss`        (Deng 2019)
  - :class:`AdaFaceLoss`        (Kim 2022)
  - :class:`CosFaceLoss`        (Wang 2018)
  - :class:`SubCenterArcFaceLoss` (Deng 2020)

Every class declares ``.family: Literal["pair", "embedding"]`` so the
training package can dispatch to the correct trainer.
"""

from bat_losses.adaface import AdaFaceLoss
from bat_losses.arcface import ArcFaceLoss
from bat_losses.bce import BCELoss
from bat_losses.cosface import CosFaceLoss
from bat_losses.focal import FocalLoss
from bat_losses.subcenter import SubCenterArcFaceLoss
from bat_losses.triplet import TripletLoss

__all__ = [
    "AdaFaceLoss",
    "ArcFaceLoss",
    "BCELoss",
    "CosFaceLoss",
    "FocalLoss",
    "SubCenterArcFaceLoss",
    "TripletLoss",
]
