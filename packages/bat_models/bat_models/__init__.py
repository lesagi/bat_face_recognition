"""bat_models — Siamese / ArcFace / AdaFace PyTorch nn.Modules."""

from bat_models.adaface import AdaFaceModel
from bat_models.arcface import ArcFaceModel
from bat_models.export import export_onnx
from bat_models.heads.adaface_head import AdaFaceHead
from bat_models.heads.arcface_head import ArcFaceHead
from bat_models.siamese import SIAMESE_INPUT_EDGE_LENGTH, SiameseModel
from bat_models.backbones.resnet import resnet50_backbone

__all__ = [
    "AdaFaceHead",
    "AdaFaceModel",
    "ArcFaceHead",
    "ArcFaceModel",
    "SIAMESE_INPUT_EDGE_LENGTH",
    "SiameseModel",
    "export_onnx",
    "resnet50_backbone",
]
