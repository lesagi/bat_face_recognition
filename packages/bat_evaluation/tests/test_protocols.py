"""End-to-end protocol smoke test using a tiny synthetic embed_fn."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bat_core import ImageRecord, Manifest  # noqa: E402

from bat_evaluation import run_eval_protocol  # noqa: E402


def _rec(identity: str, idx: int, split: str = "test") -> ImageRecord:
    return ImageRecord(
        path=Path(f"data/{identity}_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,  # type: ignore[arg-type]
        quality=0.5,
    )


def test_run_eval_protocol_with_perfect_embedder():
    # 4 identities, 3 images each. The "embedder" looks up identity from the
    # filename -> that identity's one-hot vector, so cosine similarity is 1
    # for same-id and 0 for different-id. Both verification and identification
    # should hit the metric ceiling.
    n_id = 4
    eye = np.eye(n_id, dtype=np.float32)
    id_to_vec = {f"id_{i}": eye[i] for i in range(n_id)}

    records = []
    for i in range(n_id):
        identity = f"id_{i}"
        for k in range(3):
            records.append(_rec(identity, k, split="test"))

    manifest = Manifest.from_records(records)

    def embed_fn(paths: Iterable[Path]) -> torch.Tensor:
        plist = list(paths)
        out = np.zeros((len(plist), n_id), dtype=np.float32)
        for i, p in enumerate(plist):
            ident = p.stem.split("_")[0] + "_" + p.stem.split("_")[1]
            out[i] = id_to_vec[ident]
        return torch.from_numpy(out)

    report = run_eval_protocol(
        manifest,
        split="test",
        embed_fn=embed_fn,
        gallery_size=1,
        min_probe_size=1,
        top_k=4,
    )
    assert report.identification is not None
    assert report.identification.top1 == pytest.approx(1.0)
    assert report.identification.map == pytest.approx(1.0)
    # Verification ROC-AUC should be 1.0 since same-id pairs have similarity 1
    # and different-id pairs have similarity 0 -> perfect separation.
    assert report.verification.roc_auc == pytest.approx(1.0)
