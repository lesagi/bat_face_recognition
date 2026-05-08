"""End-to-end: ``SiameseSaliencyAdapter.explain`` on a tiny pair model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bat_core.types import ImageRecord, SaliencyImage  # noqa: E402
from bat_interpretability import SiameseSaliencyAdapter  # noqa: E402


class _TinyPairModel(torch.nn.Module):
    """Minimal pair-family model — keeps tests fast."""

    family: Literal["pair", "embedding"] = "pair"

    def __init__(self, edge: int = 16) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.head = torch.nn.Linear(4 * edge * edge, 1)
        self._edge = edge

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(torch.relu(self.conv(x)))

    def forward_train(self, x_pair, labels=None):
        if isinstance(x_pair, tuple):
            a, b = x_pair
        else:
            a, b = x_pair[:, 0], x_pair[:, 1]
        emb_a = self.forward_embedding(a)
        emb_b = self.forward_embedding(b)
        return torch.sigmoid(self.head(torch.abs(emb_a - emb_b)))

    def export_for_inference(self):
        return self


def _write_fake_image(tmp_path: Path, name: str, edge: int = 16) -> Path:
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


def test_siamese_saliency_vanilla_returns_2d_map(tmp_path: Path) -> None:
    edge = 16
    model = _TinyPairModel(edge=edge)
    samples = [_record(_write_fake_image(tmp_path, "img1.png", edge=edge))]
    adapter = SiameseSaliencyAdapter(method="vanilla", input_size=edge)

    out = adapter.explain(model, samples)

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, SaliencyImage)
    assert s.method == "vanilla"
    arr = np.asarray(s.saliency)
    assert arr.ndim == 2
    assert arr.shape == (edge, edge)
    assert arr.dtype == np.float32
    # Normalised to [0, 1].
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


@pytest.mark.parametrize(
    "method", ["guided", "integrated_gradients", "smoothgrad"],
)
def test_siamese_saliency_advanced_methods(method: str, tmp_path: Path) -> None:
    edge = 16
    model = _TinyPairModel(edge=edge)
    samples = [
        _record(_write_fake_image(tmp_path, f"img_{i}.png", edge=edge), f"B{i}")
        for i in range(2)
    ]
    adapter = SiameseSaliencyAdapter(
        method=method,  # type: ignore[arg-type]
        input_size=edge,
        integration_steps=3,
        smoothing_samples=2,
    )

    out = adapter.explain(model, samples)

    assert len(out) == 2
    for s in out:
        arr = np.asarray(s.saliency)
        assert arr.shape == (edge, edge)
        assert s.method == method
