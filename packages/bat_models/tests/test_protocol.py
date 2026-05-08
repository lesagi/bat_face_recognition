"""Verify all three models satisfy ``bat_core.FaceModel`` at runtime."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bat_core.interfaces import FaceModel  # noqa: E402

from bat_models.adaface import AdaFaceModel  # noqa: E402
from bat_models.arcface import ArcFaceModel  # noqa: E402
from bat_models.siamese import SiameseModel  # noqa: E402


class _DummyBackbone(torch.nn.Module):
    output_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3 * 8 * 8, 32)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.lin(x.flatten(1))


def test_siamese_satisfies_face_model() -> None:
    model = SiameseModel()
    assert isinstance(model, FaceModel)
    assert model.family in {"pair", "embedding"}


def test_arcface_satisfies_face_model() -> None:
    model = ArcFaceModel(
        backbone=_DummyBackbone(), embedding_dim=16, num_classes=2
    )
    assert isinstance(model, FaceModel)
    assert model.family == "embedding"


def test_adaface_satisfies_face_model() -> None:
    model = AdaFaceModel(
        backbone=_DummyBackbone(), embedding_dim=16, num_classes=2
    )
    assert isinstance(model, FaceModel)
    assert model.family == "embedding"
