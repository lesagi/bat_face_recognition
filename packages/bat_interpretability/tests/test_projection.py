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
