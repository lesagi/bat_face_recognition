"""bat_stats: permutation testing + experiment-aware visualisation."""

from bat_stats.inference import (
    compute_metrics,
    partial_permute_labels,
    run_inference_permutation_test,
)
from bat_stats.naming import (
    EXPERIMENT_NAME_FIELDS,
    build_axis_label,
    build_filename,
    build_title_suffix,
    experiment_name,
    extract_components,
)
from bat_stats.permutation_test import (
    DEFAULT_METRICS,
    MetricResult,
    PermutationTest,
    PermutationTestResults,
)
from bat_stats.runner import run_inference_test, run_retrain_test
from bat_stats.trainer_hooks import TrainerProtocol, create_permutation_trainer
from bat_stats.visualizer import PermutationVisualizer

__all__ = [
    # naming
    "EXPERIMENT_NAME_FIELDS",
    "build_axis_label",
    "build_filename",
    "build_title_suffix",
    "experiment_name",
    "extract_components",
    # core test
    "DEFAULT_METRICS",
    "MetricResult",
    "PermutationTest",
    "PermutationTestResults",
    # inference path
    "compute_metrics",
    "partial_permute_labels",
    "run_inference_permutation_test",
    # visualizer
    "PermutationVisualizer",
    # orchestration
    "run_inference_test",
    "run_retrain_test",
    # trainer hooks
    "TrainerProtocol",
    "create_permutation_trainer",
]
