"""Identification-protocol tests: CMC, top-k, mAP, gallery/probe split."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bat_core import Embedding  # noqa: E402
from bat_evaluation import (  # noqa: E402
    average_precision_per_probe,
    cmc_curve,
    cosine_similarity_matrix,
    evaluate_identification,
    mean_average_precision,
)


def _make_embedding(vectors: np.ndarray, identities: list[str]) -> Embedding:
    return Embedding(
        tensor=torch.from_numpy(vectors.astype(np.float32)),
        identities=tuple(identities),
    )


def test_cmc_top1_perfect_when_gallery_matches_probes():
    # Each identity has a distinct one-hot embedding; probe = exact gallery.
    n_id = 5
    eye = np.eye(n_id)
    ids = [f"id_{i}" for i in range(n_id)]
    gallery = _make_embedding(eye, ids)
    probe = _make_embedding(eye, ids)

    metrics = evaluate_identification(gallery, probe, top_k=5)
    assert metrics.top1 == pytest.approx(1.0)
    assert metrics.top5 == pytest.approx(1.0)
    assert metrics.map == pytest.approx(1.0)


def test_cmc_is_monotonically_non_decreasing():
    rng = np.random.default_rng(1)
    n_id = 8
    dim = 16
    centers = rng.normal(0, 1, size=(n_id, dim))
    ids = [f"id_{i}" for i in range(n_id)]
    gallery_vec = centers
    probe_vec = centers + rng.normal(0, 0.5, size=(n_id, dim))

    gallery = _make_embedding(gallery_vec, ids)
    probe = _make_embedding(probe_vec, ids)
    sim = cosine_similarity_matrix(probe, gallery)
    cmc = cmc_curve(sim, list(probe.identities), list(gallery.identities), top_k=8)
    assert all(cmc[i] <= cmc[i + 1] for i in range(len(cmc) - 1))


def test_top1_top5_sanity_on_tiny_gallery():
    # 3 gallery identities, embeddings well separated; probe nearly equal to
    # one gallery row but with a small perturbation toward a wrong row.
    g = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    p = np.asarray(
        [
            [0.95, 0.10, 0.0],  # closest to id_0
            [0.10, 0.92, 0.0],  # closest to id_1
            [0.0, 0.05, 0.99],  # closest to id_2
        ]
    )
    gallery = _make_embedding(g, ["a", "b", "c"])
    probe = _make_embedding(p, ["a", "b", "c"])
    metrics = evaluate_identification(gallery, probe, top_k=3)
    assert metrics.top1 == pytest.approx(1.0)
    # CMC saturates at 1.0 by rank 3 (gallery has 3 rows).
    assert metrics.cmc[-1] == pytest.approx(1.0)


def test_map_one_when_nearest_neighbor_correct_and_unique():
    # Each identity has exactly one gallery row; the probe row points exactly
    # at its corresponding gallery row, so AP per probe = 1/1 = 1 -> mAP = 1.
    eye = np.eye(4)
    ids = [f"id_{i}" for i in range(4)]
    gallery = _make_embedding(eye, ids)
    probe = _make_embedding(eye, ids)
    sim = cosine_similarity_matrix(probe, gallery)
    map_score = mean_average_precision(sim, list(probe.identities), list(gallery.identities))
    assert map_score == pytest.approx(1.0)


def test_average_precision_intermediate_value():
    # Single probe, 4 gallery rows. The first 2 of 3 relevant are at ranks
    # 1 and 3. AP = (1/1 + 2/3) / 3.
    sim = np.asarray([0.9, 0.5, 0.8, 0.2])
    rel = np.asarray([True, True, False, True])
    # Sorted by similarity desc: ranks correspond to sim 0.9,0.8,0.5,0.2 ->
    # rel sorted = [T, F, T, T]
    # cumulative hits: [1, 1, 2, 3], precision_at_k: [1, 0.5, 2/3, 0.75]
    # picked at relevant-mask positions: 1.0, 2/3, 0.75 -> sum / 3
    expected = (1.0 + (2.0 / 3.0) + 0.75) / 3
    assert average_precision_per_probe(sim, rel) == pytest.approx(expected)


def test_cmc_short_gallery_pads_top_k():
    # gallery has 2 rows, top_k=5 -> CMC values past rank 2 stick at rank-2 value.
    g = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    p = np.asarray([[1.0, 0.0]])
    gallery = _make_embedding(g, ["a", "b"])
    probe = _make_embedding(p, ["a"])
    sim = cosine_similarity_matrix(probe, gallery)
    cmc = cmc_curve(sim, list(probe.identities), list(gallery.identities), top_k=5)
    assert cmc[0] == pytest.approx(1.0)
    assert all(cmc[i] == cmc[1] for i in range(1, 5))


def test_cosine_similarity_shape():
    g = np.random.default_rng(0).normal(size=(7, 4))
    p = np.random.default_rng(1).normal(size=(3, 4))
    gallery = _make_embedding(g, [f"g{i}" for i in range(7)])
    probe = _make_embedding(p, [f"p{i}" for i in range(3)])
    sim = cosine_similarity_matrix(probe, gallery)
    assert sim.shape == (3, 7)
    # all values in [-1, 1] since vectors are L2-normalized
    assert sim.max() <= 1 + 1e-9
    assert sim.min() >= -1 - 1e-9


def test_dim_mismatch_raises():
    g = np.zeros((2, 3))
    p = np.zeros((2, 4))
    gallery = _make_embedding(g, ["a", "b"])
    probe = _make_embedding(p, ["a", "b"])
    with pytest.raises(ValueError):
        cosine_similarity_matrix(probe, gallery)
