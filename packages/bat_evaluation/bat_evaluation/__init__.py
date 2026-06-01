"""bat_evaluation: verification + identification metrics for bat face recognition.

Public surface:

- :func:`evaluate_predictions` -> :class:`bat_core.VerificationMetrics`
- :func:`evaluate_identification` -> :class:`bat_core.IdentificationMetrics`
- :func:`optimize_youden_j` -> ``(threshold, max_J)``
- :func:`threshold_at_far` -> ``(threshold, recall)`` at a target FAR budget
- :func:`neg_log_1_minus`, :func:`arccos_scale` -- monotone score
  transformations for visualising cosine-similarity scores; semantics-free.
- :func:`run_eval_protocol` -> :class:`bat_core.EvalReport`
- :func:`split_gallery_probe`, :func:`split_manifest_for_identification`
- :func:`predictions_from_embedding`, :func:`cosine_similarity_matrix`,
  :func:`tar_at_far`, :func:`confusion_at_threshold`

Outputs are always ``bat_core`` dataclasses; this package never writes CSVs.
"""

from bat_evaluation.gallery_probe import (
    EmbedFn,
    GalleryProbeSplit,
    materialize_embedding,
    split_gallery_probe,
    split_manifest_for_identification,
)
from bat_evaluation.identification import (
    average_precision_per_probe,
    cmc_curve,
    cosine_similarity_matrix,
    evaluate_identification,
    mean_average_precision,
    tar_at_far_from_gallery_probe,
)
from bat_evaluation.protocols import run_eval_protocol, run_identification, run_verification
from bat_evaluation.scaling import arccos_scale, inv_neg_log_1_minus, neg_log_1_minus
from bat_evaluation.threshold import optimize_youden_j, threshold_at_far
from bat_evaluation.verification import (
    ConfusionAtThreshold,
    compute_roc,
    confusion_at_threshold,
    evaluate_predictions,
    predictions_from_embedding,
    tar_at_far,
)

__all__ = [
    "ConfusionAtThreshold",
    "EmbedFn",
    "GalleryProbeSplit",
    "average_precision_per_probe",
    "cmc_curve",
    "compute_roc",
    "confusion_at_threshold",
    "cosine_similarity_matrix",
    "evaluate_identification",
    "evaluate_predictions",
    "materialize_embedding",
    "mean_average_precision",
    "optimize_youden_j",
    "predictions_from_embedding",
    "run_eval_protocol",
    "run_identification",
    "run_verification",
    "split_gallery_probe",
    "split_manifest_for_identification",
    "arccos_scale",
    "inv_neg_log_1_minus",
    "neg_log_1_minus",
    "tar_at_far",
    "tar_at_far_from_gallery_probe",
    "threshold_at_far",
]
