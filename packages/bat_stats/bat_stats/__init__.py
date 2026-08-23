"""bat_stats: permutation testing + experiment-aware visualisation."""

from bat_stats.inference import (
    compute_metrics,
    partial_permute_labels,
    run_inference_permutation_test,
)
from bat_stats.model_comparison import (
    BootstrapResult,
    ComparisonResult,
    EquivalenceResult,
    TTestResult,
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    combine_pvalues,
    compare_groups,
    corrected_resampled_ttest,
    harmonic_mean_p,
    hedges_g,
    hierarchical_bootstrap,
    interpret_delta,
    permutation_p,
    permutation_p_floor,
    tost,
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
from bat_stats.test_curves import (
    cmc_curve_figure,
    recall_vs_far_figure,
    roc_curve_figure,
    save_test_curves,
)
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
    # model / fold / species comparison
    "BootstrapResult",
    "ComparisonResult",
    "EquivalenceResult",
    "TTestResult",
    "benjamini_hochberg",
    "bootstrap_ci",
    "cliffs_delta",
    "combine_pvalues",
    "compare_groups",
    "corrected_resampled_ttest",
    "harmonic_mean_p",
    "hedges_g",
    "hierarchical_bootstrap",
    "interpret_delta",
    "permutation_p",
    "permutation_p_floor",
    "tost",
    # inference path
    "compute_metrics",
    "partial_permute_labels",
    "run_inference_permutation_test",
    # visualizer
    "PermutationVisualizer",
    # test curves
    "cmc_curve_figure",
    "recall_vs_far_figure",
    "roc_curve_figure",
    "save_test_curves",
    # orchestration
    "run_inference_test",
    "run_retrain_test",
    # trainer hooks
    "TrainerProtocol",
    "create_permutation_trainer",
]
