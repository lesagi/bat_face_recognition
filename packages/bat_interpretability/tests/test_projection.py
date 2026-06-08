"""Tests for ``EmbeddingProjectionAdapter``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from bat_core.types import Embedding  # noqa: E402
from bat_interpretability import EmbeddingProjectionAdapter  # noqa: E402


def _fixture_embedding(n: int = 20, d: int = 8) -> Embedding:
    rng = np.random.default_rng(seed=0)
    arr = rng.standard_normal((n, d)).astype(np.float32)
    tensor = torch.from_numpy(arr)
    identities = tuple(f"id_{i % 4}" for i in range(n))
    return Embedding(tensor=tensor, identities=identities)


def test_project_embeddings_tsne_only(tmp_path: Path) -> None:
    emb = _fixture_embedding(n=20)
    adapter = EmbeddingProjectionAdapter(method="tsne", output_dir=tmp_path)
    out = adapter.project_embeddings(emb)

    assert len(out) == 1
    s = out[0]
    coords = np.asarray(s.saliency)
    assert coords.shape == (20, 2)
    assert s.method == "tsne"
    assert Path(s.image_path).exists()
    assert Path(s.image_path).suffix == ".png"


def test_project_embeddings_umap_only(tmp_path: Path) -> None:
    pytest.importorskip("umap")
    emb = _fixture_embedding(n=20)
    adapter = EmbeddingProjectionAdapter(method="umap", output_dir=tmp_path)
    out = adapter.project_embeddings(emb)

    assert len(out) == 1
    s = out[0]
    coords = np.asarray(s.saliency)
    assert coords.shape == (20, 2)
    assert s.method == "umap"


def test_project_embeddings_both(tmp_path: Path) -> None:
    pytest.importorskip("umap")
    emb = _fixture_embedding(n=20)
    adapter = EmbeddingProjectionAdapter(method="both", output_dir=tmp_path)
    out = adapter.project_embeddings(emb)

    assert len(out) == 2
    methods = {s.method for s in out}
    assert methods == {"tsne", "umap"}


def test_scatter_legend_includes_per_identity_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legend labels must read ``<identity> (n=<count>)`` per the reporting fix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: dict[str, list[str]] = {}
    real_close = plt.close

    def _capture_close(fig: object = None) -> None:
        if fig is not None and getattr(fig, "axes", None):
            legend = fig.axes[0].get_legend()
            if legend is not None:
                captured["labels"] = [t.get_text() for t in legend.get_texts()]
        real_close(fig)

    monkeypatch.setattr(plt, "close", _capture_close)

    adapter = EmbeddingProjectionAdapter(method="tsne", output_dir=tmp_path)
    coords = np.zeros((6, 2), dtype=np.float32)
    identities = ["A", "A", "A", "B", "B", "C"]
    adapter._save_scatter(coords, identities, tmp_path, "tsne")

    labels = captured.get("labels", [])
    assert "A (n=3)" in labels
    assert "B (n=2)" in labels
    assert "C (n=1)" in labels
