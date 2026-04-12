"""
Inference-Based Permutation Test for Model Significance.

Instead of retraining from scratch N times (expensive), this module:
1. Loads a trained model and runs inference once on the test set
2. Computes observed metrics from predictions vs real labels
3. Builds a null distribution by permuting test labels N times and
   recomputing metrics each time (pure numpy -- no GPU needed)
4. Computes p-values with the same formula as the classical approach

This tests: "Are the model's predictions significantly better correlated
with the true labels than with random labels?"

Optional degradation curve: evaluates metrics at increasing fractions
of label permutation (0%, 5%, 10%, ..., 100%) to show how performance
degrades as labels become more randomized.
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .permutation_test import MetricResult, PermutationTest, PermutationTestResults


def _parse_experiment_params(experiment_dir: str) -> Dict[str, str]:
    """Extract training params from the experiment directory name.

    Raises ValueError if the directory name doesn't match the expected format.
    """
    dirname = os.path.basename(experiment_dir)
    m = re.match(
        r"\d{8}_siamese_(?P<species>rousettus|mauritius)"
        r"_(?P<source>video|still)"
        r"_(?P<aug>augmented|no_aug)"
        r"_(?P<bg>\w+)_bg_",
        dirname,
    )
    if not m:
        raise ValueError(
            f"Cannot parse training params from directory name: {dirname}. "
            f"Expected format: YYYYMMDD_siamese_{{species}}_{{source}}_{{aug}}_{{bg}}_bg_{{run_id}}"
        )
    species = m.group("species")
    return {
        "bat_type": "m" if species == "mauritius" else "r",
        "species": species,
        "data_source": m.group("source"),
        "augmented": m.group("aug") == "augmented",
        "background": m.group("bg"),
    }


def _load_optimal_threshold(experiment_dir: str) -> float:
    """Load the optimal classification threshold from training_summary.json."""
    summary_path = os.path.join(experiment_dir, "training_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            if "optimal_threshold" in summary:
                return summary["optimal_threshold"]["value"]
        except Exception:
            pass
    return 0.5


def _compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute F1, accuracy, precision, recall from predictions and labels."""
    binary_preds = (predictions >= threshold).astype(np.float32)

    tp = np.sum((binary_preds == 1) & (labels == 1))
    fp = np.sum((binary_preds == 1) & (labels == 0))
    fn = np.sum((binary_preds == 0) & (labels == 1))
    tn = np.sum((binary_preds == 0) & (labels == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }


def _partial_permute_labels(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Permute a fraction of labels while keeping the rest fixed.

    At fraction=0.0 labels are unchanged; at 1.0 all labels are shuffled.
    The label distribution (count of 0s and 1s) is always preserved.
    """
    if fraction <= 0.0:
        return labels.copy()
    n = len(labels)
    k = max(1, int(round(n * fraction)))
    result = labels.copy()
    indices = rng.choice(n, size=k, replace=False)
    subset = result[indices].copy()
    rng.shuffle(subset)
    result[indices] = subset
    return result


def run_inference_permutation_test(
    experiment_dir: str,
    model_path: str,
    n_permutations: int = 1000,
    significance_level: float = 0.05,
    metrics_to_test: Optional[List[str]] = None,
    degradation_fractions: Optional[List[float]] = None,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    gpu: Optional[int] = None,
) -> Tuple[PermutationTestResults, Optional[Dict[str, List[Tuple[float, float]]]]]:
    """Run an inference-based permutation test on a trained model.

    Args:
        experiment_dir: Path to the experiment directory (contains config_snapshot.yml,
            training_summary.json, best_model_* subdirs).
        model_path: Path to the saved model directory to load.
        n_permutations: Number of label permutations for the null distribution.
        significance_level: Alpha threshold for significance.
        metrics_to_test: Which metrics to test (default: f1, accuracy, precision, recall).
        degradation_fractions: If provided, compute metrics at each fraction of label
            permutation (e.g. [0.0, 0.1, 0.2, ..., 1.0]). None to skip.
        verbose: Print progress.
        output_dir: Where to save results. Defaults to {experiment_dir}/permutation_test/.
        gpu: GPU device ID. If None, uses default.

    Returns:
        (results, degradation_data) where degradation_data is None if not requested,
        or a dict mapping metric names to lists of (fraction, mean_metric_value) tuples.
    """
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if metrics_to_test is None:
        metrics_to_test = ["f1", "accuracy", "precision", "recall"]

    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(experiment_dir, f'permutation_test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Parse experiment config
    # ------------------------------------------------------------------
    exp_params = _parse_experiment_params(experiment_dir)
    threshold = _load_optimal_threshold(experiment_dir)

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("Inference-Based Permutation Test", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Experiment:   {os.path.basename(experiment_dir)}", flush=True)
        print(f"  Model:        {model_path}", flush=True)
        print(f"  Species:      {exp_params['species']}", flush=True)
        print(f"  Data source:  {exp_params['data_source']}", flush=True)
        print(f"  Background:   {exp_params['background']}", flush=True)
        print(f"  Threshold:    {threshold:.4f}", flush=True)
        print(f"  Permutations: {n_permutations}", flush=True)
        print(f"  Output:       {output_dir}", flush=True)
        print(f"{'='*70}\n", flush=True)

    # ------------------------------------------------------------------
    # 2. Load model (delays TF import until here)
    # ------------------------------------------------------------------
    if verbose:
        print("Loading model...", flush=True)

    from generate_predictions import load_siamese_model
    model = load_siamese_model(model_path)
    model_input_size = model.input_shape[0][1]

    # ------------------------------------------------------------------
    # 3. Build test dataset (same config as original training)
    # ------------------------------------------------------------------
    if verbose:
        print("Building test dataset...", flush=True)

    import tensorflow as tf
    from config.loader import load_config
    from siamese_data.data_splitter import SiameseNetworkTrainingDataSplitter

    cfg = load_config()
    sn_train = cfg.siamese_network.training
    input_paths = cfg.siamese_network.input_paths[exp_params['species']]

    bg_key_map = {"green": "green_bg_input", "random": "random_bg_input", "original": "original_bg_input"}
    input_dir = input_paths.get(bg_key_map[exp_params['background']])
    if not input_dir or not os.path.exists(input_dir):
        raise ValueError(
            f"Input directory not found for background '{exp_params['background']}': {input_dir}"
        )

    training_portion = sn_train.get("train_val_split", 0.7)
    pair_mode = sn_train.get("pair_mode", "combination")

    splitter = SiameseNetworkTrainingDataSplitter(
        [input_dir],
        training_portion=training_portion,
        mode=pair_mode,
        permute_labels=False,
        split_mode="image_split",
    )

    if splitter.test_data is None:
        raise ValueError("Data splitter returned no test data")

    # ------------------------------------------------------------------
    # 4. Run inference on all test pairs
    # ------------------------------------------------------------------
    if verbose:
        print("Running inference on test set...", flush=True)

    from generate_predictions import preprocess_siamese_input_flexible

    all_predictions = []
    all_labels = []
    batch_count = 0
    BATCH_SIZE = 128

    test_batches = splitter.test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for batch in test_batches:
        img1_paths, img2_paths, labels, _class_info = batch
        batch_img1 = []
        batch_img2 = []
        batch_labels = []

        for i in range(len(labels)):
            p1 = img1_paths[i].numpy()
            p2 = img2_paths[i].numpy()
            if isinstance(p1, bytes):
                p1 = p1.decode('utf-8')
            if isinstance(p2, bytes):
                p2 = p2.decode('utf-8')

            img1 = preprocess_siamese_input_flexible(p1, target_size=model_input_size)
            img2 = preprocess_siamese_input_flexible(p2, target_size=model_input_size)
            batch_img1.append(img1)
            batch_img2.append(img2)
            batch_labels.append(float(labels[i].numpy()))

        if not batch_img1:
            continue

        batch_img1_t = tf.stack(batch_img1)
        batch_img2_t = tf.stack(batch_img2)
        preds = model.predict([batch_img1_t, batch_img2_t], verbose=0)
        all_predictions.extend(preds.flatten().tolist())
        all_labels.extend(batch_labels)

        batch_count += 1
        if verbose and batch_count % 20 == 0:
            print(f"  Processed {batch_count} batches ({len(all_labels)} pairs)...", flush=True)

    predictions = np.array(all_predictions, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    if verbose:
        print(f"  Total test pairs: {len(labels)}", flush=True)
        pos_count = int(np.sum(labels == 1))
        print(f"  Label distribution: {pos_count} positive, {len(labels) - pos_count} negative", flush=True)

    # ------------------------------------------------------------------
    # 5. Compute observed metrics
    # ------------------------------------------------------------------
    observed_metrics = _compute_metrics(predictions, labels, threshold)

    if verbose:
        print(f"\nObserved metrics:", flush=True)
        for k, v in observed_metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 6. Build null distribution (pure numpy, no GPU)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nBuilding null distribution ({n_permutations} permutations)...", flush=True)

    rng = np.random.default_rng(seed=42)
    null_distributions: Dict[str, List[float]] = {m: [] for m in metrics_to_test}

    for i in range(n_permutations):
        permuted = _partial_permute_labels(labels, fraction=1.0, rng=rng)
        perm_metrics = _compute_metrics(predictions, permuted, threshold)

        for metric in metrics_to_test:
            if metric in perm_metrics:
                null_distributions[metric].append(perm_metrics[metric])

        if verbose and (i + 1) % max(1, n_permutations // 10) == 0:
            print(f"  Permutation {i+1}/{n_permutations}", flush=True)

    # ------------------------------------------------------------------
    # 7. Compute p-values and build results
    # ------------------------------------------------------------------
    perm_test = PermutationTest(
        n_permutations=n_permutations,
        permutation_epochs=0,
        significance_level=significance_level,
        metrics_to_test=metrics_to_test,
        save_null_distribution=True,
    )

    results_metrics: Dict[str, MetricResult] = {}
    for metric in metrics_to_test:
        if metric not in observed_metrics:
            continue
        observed = observed_metrics[metric]
        null_dist = np.array(null_distributions[metric])
        p_value = perm_test.compute_p_value(observed, null_dist)
        null_mean = float(np.mean(null_dist))
        null_std = float(np.std(null_dist))

        results_metrics[metric] = MetricResult(
            observed=observed,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            significant=p_value < significance_level,
            null_distribution=null_distributions[metric],
        )

    total_time = time.time() - start_time

    results = PermutationTestResults(
        n_permutations=n_permutations,
        significance_level=significance_level,
        permutation_epochs=0,
        metrics=results_metrics,
        total_time_seconds=total_time,
    )

    # ------------------------------------------------------------------
    # 8. Optional degradation curve
    # ------------------------------------------------------------------
    degradation_data = None
    if degradation_fractions is not None:
        if verbose:
            print(f"\nComputing degradation curve ({len(degradation_fractions)} fractions)...", flush=True)

        degradation_data = {m: [] for m in metrics_to_test}
        n_repeats = min(50, max(10, n_permutations // 20))

        for frac in degradation_fractions:
            frac_metrics = {m: [] for m in metrics_to_test}

            for _ in range(n_repeats):
                permuted = _partial_permute_labels(labels, fraction=frac, rng=rng)
                m_vals = _compute_metrics(predictions, permuted, threshold)
                for metric in metrics_to_test:
                    if metric in m_vals:
                        frac_metrics[metric].append(m_vals[metric])

            for metric in metrics_to_test:
                mean_val = float(np.mean(frac_metrics[metric])) if frac_metrics[metric] else 0.0
                degradation_data[metric].append((frac, mean_val))

            if verbose:
                vals_str = ", ".join(
                    f"{m}: {np.mean(frac_metrics[m]):.4f}" for m in metrics_to_test if frac_metrics[m]
                )
                print(f"  fraction={frac:.2f}: {vals_str}", flush=True)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, 'permutation_results.json')
    results.save(results_path)
    if verbose:
        print(f"\nResults saved to: {results_path}", flush=True)

    if degradation_data:
        deg_path = os.path.join(output_dir, 'degradation_curve.json')
        with open(deg_path, 'w') as f:
            json.dump(degradation_data, f, indent=2)
        if verbose:
            print(f"Degradation curve saved to: {deg_path}", flush=True)

    if verbose:
        results.print_summary()

    return results, degradation_data
