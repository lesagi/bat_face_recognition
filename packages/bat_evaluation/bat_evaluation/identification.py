"""Open-set identification protocol: CMC, top-k accuracy, mAP, TAR@FAR.

Given a gallery (one or more enrolled embeddings per identity) and a probe
set (query embeddings with known identities), compute:

- The CMC curve up to ``top_k`` ranks (length ``top_k``).
- Top-1 and top-5 identification accuracy.
- Mean Average Precision (mAP) treating each probe as an information-retrieval
  query against the gallery.
- TAR@FAR at 1e-3 and 1e-4, derived from the full probe-vs-gallery similarity
  matrix. Each (probe, gallery) cell is one verification trial: positive if
  the identities match, negative otherwise.

The output is a :class:`bat_core.IdentificationMetrics` dataclass.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from bat_core import Embedding, IdentificationMetrics

from bat_evaluation.verification import compute_roc, tar_at_far


def _embedding_to_numpy(embedding: Embedding) -> NDArray[np.float64]:
    import torch

    tensor = embedding.tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Embedding.tensor must be a torch.Tensor")
    return tensor.detach().to(dtype=torch.float32, device="cpu").numpy().astype(np.float64)


def _l2_normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def cosine_similarity_matrix(
    probe: Embedding | NDArray[np.float64],
    gallery: Embedding | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the cosine-similarity matrix between probe and gallery.

    Output shape: ``(n_probe, n_gallery)``.
    """
    p = (
        _embedding_to_numpy(probe)
        if isinstance(probe, Embedding)
        else np.asarray(probe, dtype=np.float64)
    )
    g = (
        _embedding_to_numpy(gallery)
        if isinstance(gallery, Embedding)
        else np.asarray(gallery, dtype=np.float64)
    )
    if p.ndim != 2 or g.ndim != 2:
        raise ValueError("probe and gallery embeddings must be 2-D")
    if p.shape[1] != g.shape[1]:
        raise ValueError(
            f"probe and gallery have different embedding dim: "
            f"{p.shape[1]} vs {g.shape[1]}"
        )
    p = _l2_normalize(p)
    g = _l2_normalize(g)
    return p @ g.T


def cmc_curve(
    similarity: NDArray[np.float64],
    probe_ids: Iterable[str],
    gallery_ids: Iterable[str],
    top_k: int,
) -> NDArray[np.float64]:
    """Cumulative match characteristic curve, ranks 1..top_k.

    For each probe, gallery rows are sorted by similarity in decreasing order;
    a "hit at rank k" means at least one gallery entry sharing the probe's
    identity appears within the top-k. The CMC value at rank k is the
    fraction of probes with a hit at rank k or earlier; it is monotonically
    non-decreasing in k.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    probes = list(probe_ids)
    gals = list(gallery_ids)
    if similarity.shape != (len(probes), len(gals)):
        raise ValueError(
            f"similarity shape {similarity.shape} != "
            f"(n_probe={len(probes)}, n_gallery={len(gals)})"
        )
    if len(probes) == 0:
        return np.zeros(top_k, dtype=np.float64)

    cap = min(top_k, len(gals))
    # argsort descending: take top-cap columns per row
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :cap]
    gallery_arr = np.asarray(gals, dtype=object)
    probe_arr = np.asarray(probes, dtype=object)

    ranked_ids = gallery_arr[order]  # shape (n_probe, cap)
    matches = ranked_ids == probe_arr[:, None]
    # First hit rank per probe (1-indexed); -1 if no hit in top-cap
    has_hit = matches.any(axis=1)
    first_hit = np.where(has_hit, matches.argmax(axis=1), -1)

    cmc = np.zeros(top_k, dtype=np.float64)
    n_probe = len(probes)
    for k in range(top_k):
        if k < cap:
            cmc[k] = float(((first_hit >= 0) & (first_hit <= k)).sum()) / n_probe
        else:
            # past gallery size: value sticks at the rank-cap value
            cmc[k] = cmc[k - 1] if k > 0 else 0.0
    return cmc


def average_precision_per_probe(
    similarity_row: NDArray[np.float64],
    relevant_mask: NDArray[np.bool_],
) -> float:
    """Average precision for a single probe (information-retrieval style).

    ``similarity_row`` is a 1-D vector of similarities to the gallery;
    ``relevant_mask`` is the same length and True where the gallery row
    shares the probe's identity.
    """
    n_relevant = int(relevant_mask.sum())
    if n_relevant == 0:
        return 0.0
    order = np.argsort(-similarity_row, kind="stable")
    sorted_rel = relevant_mask[order]
    cum_hits = np.cumsum(sorted_rel.astype(np.float64))
    ranks = np.arange(1, sorted_rel.size + 1, dtype=np.float64)
    precision_at_k = cum_hits / ranks
    return float(precision_at_k[sorted_rel].sum() / n_relevant)


def mean_average_precision(
    similarity: NDArray[np.float64],
    probe_ids: Iterable[str],
    gallery_ids: Iterable[str],
) -> float:
    """Mean Average Precision over all probes."""
    probes = list(probe_ids)
    gals = list(gallery_ids)
    if similarity.shape != (len(probes), len(gals)):
        raise ValueError(
            f"similarity shape {similarity.shape} != "
            f"(n_probe={len(probes)}, n_gallery={len(gals)})"
        )
    if not probes:
        return 0.0
    gallery_arr = np.asarray(gals, dtype=object)
    aps: list[float] = []
    for i, pid in enumerate(probes):
        rel = gallery_arr == pid
        aps.append(average_precision_per_probe(similarity[i], rel))
    return float(np.mean(aps))


def evaluate_identification(
    gallery: Embedding,
    probe: Embedding,
    top_k: int = 10,
) -> IdentificationMetrics:
    """Compute open-set identification metrics from gallery + probe embeddings.

    Returns a :class:`bat_core.IdentificationMetrics` containing:
    top-1 accuracy, top-5 accuracy, mAP, and the CMC curve up to ``top_k``.
    """
    sim = cosine_similarity_matrix(probe, gallery)
    cmc = cmc_curve(sim, probe.identities, gallery.identities, top_k=top_k)
    top1 = float(cmc[0]) if top_k >= 1 else 0.0
    top5 = float(cmc[min(4, top_k - 1)]) if top_k >= 1 else 0.0
    map_score = mean_average_precision(sim, probe.identities, gallery.identities)
    return IdentificationMetrics(
        top1=top1,
        top5=top5,
        map=map_score,
        cmc=tuple(float(v) for v in cmc),
    )


def tar_at_far_from_gallery_probe(
    gallery: Embedding,
    probe: Embedding,
    target_far: float,
) -> float:
    """TAR@FAR derived from the full probe-vs-gallery similarity matrix.

    Each cell of the similarity matrix is one verification trial; positive if
    the probe and gallery share an identity, negative otherwise.
    """
    sim = cosine_similarity_matrix(probe, gallery)
    p_ids = np.asarray(list(probe.identities), dtype=object)
    g_ids = np.asarray(list(gallery.identities), dtype=object)
    labels = (p_ids[:, None] == g_ids[None, :]).astype(np.int64)
    y_true = labels.ravel()
    y_score = sim.ravel()
    _, fpr, tpr, _ = compute_roc(y_true, y_score)
    return tar_at_far(fpr, tpr, target_far)


__all__ = [
    "average_precision_per_probe",
    "cmc_curve",
    "cosine_similarity_matrix",
    "evaluate_identification",
    "mean_average_precision",
    "tar_at_far_from_gallery_probe",
]
