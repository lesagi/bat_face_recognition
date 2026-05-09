"""End-to-end: ``GradCAMAdapter.explain`` on a tiny conv-backbone model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bat_core.types import ImageRecord, SaliencyImage  # noqa: E402
from bat_interpretability import GradCAMAdapter  # noqa: E402


class _TinyConvBackbone(torch.nn.Module):
    """Two conv layers + GAP -> 8-d feature vector."""

    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return x


class _TinyEmbeddingModel(torch.nn.Module):
    """Mimics ``ArcFaceModel`` API but with a 8-d backbone."""

    family: Literal["pair", "embedding"] = "embedding"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyConvBackbone()
        self.projection = torch.nn.Linear(8, 4)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return torch.nn.functional.normalize(self.projection(feat), p=2, dim=1)

    def forward_train(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.projection(feat)  # arbitrary "logits"

    def export_for_inference(self):
        return self


def _write_fake_image(tmp_path: Path, name: str, edge: int = 32) -> Path:
    from PIL import Image

    img = (np.random.rand(edge, edge, 3) * 255).astype(np.uint8)
    p = tmp_path / name
    Image.fromarray(img).save(p)
    return p


def _record(path: Path, identity: str = "B1") -> ImageRecord:
    return ImageRecord(
        path=path,
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split="val",
        quality=0.9,
    )


def test_gradcam_embedding_target_returns_2d_heatmap(tmp_path: Path) -> None:
    edge = 32
    model = _TinyEmbeddingModel()
    samples = [_record(_write_fake_image(tmp_path, "img.png", edge=edge))]
    adapter = GradCAMAdapter(target="embedding_magnitude", input_size=edge)

    out = adapter.explain(model, samples)

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, SaliencyImage)
    arr = np.asarray(s.saliency)
    # Adapter resizes the cam back to input H, W.
    assert arr.ndim == 2
    assert arr.shape == (edge, edge)
    assert s.method.startswith("gradcam_")


def test_gradcam_class_target(tmp_path: Path) -> None:
    edge = 32
    model = _TinyEmbeddingModel()
    samples = [_record(_write_fake_image(tmp_path, "img.png", edge=edge))]
    adapter = GradCAMAdapter(target="class", target_class=0, input_size=edge)

    out = adapter.explain(model, samples)
    assert len(out) == 1
    arr = np.asarray(out[0].saliency)
    assert arr.shape == (edge, edge)


def test_gradcam_explicit_target_layer(tmp_path: Path) -> None:
    edge = 32
    model = _TinyEmbeddingModel()
    samples = [_record(_write_fake_image(tmp_path, "img.png", edge=edge))]
    adapter = GradCAMAdapter(
        target="embedding_magnitude",
        input_size=edge,
        target_layer_name="backbone.conv1",
    )
    out = adapter.explain(model, samples)
    arr = np.asarray(out[0].saliency)
    assert arr.shape == (edge, edge)
