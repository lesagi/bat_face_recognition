"""Orchestration: run verification + identification end-to-end.

This module is intentionally model-agnostic. It accepts:

- a :class:`bat_core.Manifest`,
- a target split (``"val"`` or ``"test"``),
- an embedding callable ``embed_fn(paths) -> torch.Tensor``,

and returns a :class:`bat_core.EvalReport` containing the verification +
identification metrics. We do not import ``torch.nn`` here so this package
remains useful for any model that can produce embeddings.

Verification predictions are derived from the same gallery + probe pool by
forming all unordered pairs over the union of gallery and probe records,
labelled 1 when the two records share an identity. Cosine similarity is the
score function.
"""
from __future__ import annotations

import numpy as np

from bat_core import (
    Embedding,
    EvalReport,
    IdentificationMetrics,
    Manifest,
    Predictions,
    Split,
    VerificationMetrics,
)

from bat_evaluation.gallery_probe import (
    EmbedFn,
    GalleryProbeSplit,
    materialize_embedding,
    split_manifest_for_identification,
)
from bat_evaluation.identification import (
    cosine_similarity_matrix,
    evaluate_identification,
)
from bat_evaluation.verification import (
    evaluate_predictions,
    predictions_from_embedding,
)


def run_verification(embedding: Embedding) -> tuple[VerificationMetrics, Predictions]:
    """Verification metrics from all unordered pairs of an embedding."""
    predictions = predictions_from_embedding(embedding)
    metrics = evaluate_predictions(predictions)
    return metrics, predictions


def run_identification(
    gallery: Embedding,
    probe: Embedding,
    top_k: int = 10,
) -> IdentificationMetrics:
    """Identification metrics for the given gallery + probe."""
    return evaluate_identification(gallery, probe, top_k=top_k)


def run_eval_protocol(
    manifest: Manifest,
    split: Split,
    embed_fn: EmbedFn,
    *,
    gallery_size: int = 1,
    min_probe_size: int = 1,
    top_k: int = 10,
) -> EvalReport:
    """Run the full verification + identification protocol on a manifest split.

    Parameters
    ----------
    manifest: source manifest.
    split: ``"train"``, ``"val"``, or ``"test"``. Identification needs >= 2
        images per identity, so the train split rarely makes sense here.
    embed_fn: callable that maps a list of image paths to a 2-D
        ``torch.Tensor`` of embeddings (shape ``(N, D)``).
    gallery_size: number of images per identity assigned to the gallery.
    min_probe_size: minimum number of probe images per identity.
    top_k: number of CMC ranks to compute.
    """
    gp: GalleryProbeSplit = split_manifest_for_identification(
        manifest,
        split=split,
        gallery_size=gallery_size,
        min_probe_size=min_probe_size,
    )
    gallery = materialize_embedding(gp.gallery_records, embed_fn)
    probe = materialize_embedding(gp.probe_records, embed_fn)

    identification = run_identification(gallery, probe, top_k=top_k)

    # Verification: cosine similarity between every probe and every gallery row,
    # labelled by identity match. This avoids the train-time pair construction
    # so the result is directly comparable across model families.
    sim = cosine_similarity_matrix(probe, gallery)
    p_ids = np.asarray(list(probe.identities), dtype=object)
    g_ids = np.asarray(list(gallery.identities), dtype=object)
    labels = (p_ids[:, None] == g_ids[None, :]).astype(np.int64).ravel()
    scores = sim.ravel()
    predictions = Predictions(
        y_true=tuple(int(v) for v in labels),
        y_score=tuple(float(v) for v in scores),
    )
    verification = evaluate_predictions(predictions)

    return EvalReport(
        verification=verification,
        identification=identification,
        predictions=predictions,
    )


__all__ = [
    "run_eval_protocol",
    "run_identification",
    "run_verification",
]
