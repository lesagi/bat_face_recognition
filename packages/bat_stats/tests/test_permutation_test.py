"""Numerical / behavioural tests for the core PermutationTest algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from bat_stats.permutation_test import (
    DEFAULT_METRICS,
    MetricResult,
    PermutationTest,
    PermutationTestResults,
)


def test_compute_p_value_continuity_correction() -> None:
    pt = PermutationTest(verbose=False)
    null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # 0 of 5 null values >= 0.9 → (0+1)/(5+1) = 1/6
    assert pt.compute_p_value(0.9, null) == pytest.approx(1 / 6)
    # 5 of 5 null values >= 0.05 → (5+1)/(5+1) = 1.0
    assert pt.compute_p_value(0.05, null) == pytest.approx(1.0)
    # 3 of 5 null values >= 0.3 → (3+1)/(5+1) = 4/6
    assert pt.compute_p_value(0.3, null) == pytest.approx(4 / 6)


def test_run_from_null_distributions_known_pvalue() -> None:
    """Reproducible p-value on a synthetic null distribution.

    With seed=42 we draw 200 N(0.5, 0.05) samples for the f1 null and check the
    observed=0.95 (way out in the right tail) yields p = 1/(200+1).
    """

    rng = np.random.default_rng(42)
    null_f1 = rng.normal(0.5, 0.05, size=200).tolist()
    pt = PermutationTest(
        n_permutations=200,
        permutation_epochs=0,
        metrics_to_test=["f1"],
        verbose=False,
    )
    results = pt.run_from_null_distributions(
        observed_metrics={"f1": 0.95},
        null_distributions={"f1": null_f1},
    )
    assert "f1" in results.metrics
    f1_result = results.metrics["f1"]
    assert f1_result.observed == 0.95
    # All 200 null samples are well below 0.95 → p = 1/201
    assert f1_result.p_value == pytest.approx(1 / 201)
    assert f1_result.significant is True
    # Null distribution of N(0.5, 0.05) → mean ≈ 0.5, std ≈ 0.05
    assert 0.48 < f1_result.null_mean < 0.52
    assert 0.03 < f1_result.null_std < 0.07


def test_observed_inside_null_is_not_significant() -> None:
    rng = np.random.default_rng(7)
    null = rng.normal(0.5, 0.1, size=500).tolist()
    pt = PermutationTest(
        n_permutations=500,
        metrics_to_test=["f1"],
        verbose=False,
    )
    results = pt.run_from_null_distributions(
        observed_metrics={"f1": 0.5},
        null_distributions={"f1": null},
    )
    f1 = results.metrics["f1"]
    assert f1.p_value > 0.05
    assert f1.significant is False


def test_run_with_mock_trainer_factory() -> None:
    """End-to-end ``run`` with a mock trainer (no torch)."""

    class _MockTrainer:
        def __init__(self) -> None:
            self.calls = 0

        def reset_for_new_permutation(self) -> None:
            self.calls += 1

        def train_and_evaluate(self) -> dict[str, float]:
            return {"f1": 0.5, "accuracy": 0.55, "precision": 0.5, "recall": 0.5}

    instances: list[_MockTrainer] = []

    def factory(**_kwargs: object) -> _MockTrainer:
        t = _MockTrainer()
        instances.append(t)
        return t

    pt = PermutationTest(n_permutations=5, permutation_epochs=1, verbose=False)
    results = pt.run(
        observed_metrics={"f1": 0.95, "accuracy": 0.95, "precision": 0.95, "recall": 0.95},
        trainer_factory=factory,
    )
    # Trainer should be created exactly once (model reuse semantic from legacy).
    assert len(instances) == 1
    # reset_for_new_permutation called n-1 times.
    assert instances[0].calls == 4
    # Null distributions populated.
    for m in DEFAULT_METRICS:
        assert m in results.metrics
        assert len(results.metrics[m].null_distribution) == 5
    # Observed beats every null sample → p = 1/6
    assert results.metrics["f1"].p_value == pytest.approx(1 / 6)


def test_results_save_and_load_roundtrip(tmp_path) -> None:
    results = PermutationTestResults(
        n_permutations=10,
        significance_level=0.05,
        permutation_epochs=0,
        metrics={
            "f1": MetricResult(
                observed=0.9,
                null_mean=0.5,
                null_std=0.05,
                p_value=0.01,
                significant=True,
                null_distribution=[0.4, 0.5, 0.6],
            )
        },
        total_time_seconds=1.2,
    )
    path = tmp_path / "perm.json"
    results.save(path)
    loaded = PermutationTestResults.load(path)
    assert loaded.n_permutations == 10
    assert loaded.metrics["f1"].observed == 0.9
    assert loaded.metrics["f1"].null_distribution == [0.4, 0.5, 0.6]


def test_default_metrics_constant() -> None:
    assert "f1" in DEFAULT_METRICS
    assert set(DEFAULT_METRICS) == {"f1", "accuracy", "precision", "recall"}
