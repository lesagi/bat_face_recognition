"""Tests for the orchestration runner."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from bat_stats.naming import experiment_name  # noqa: E402
from bat_stats.runner import run_inference_test, run_retrain_test  # noqa: E402


def test_run_inference_test_writes_experiment_aware_files(tmp_path: Path, cfg) -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    predictions = labels.copy().astype(np.float32)

    results = run_inference_test(
        cfg=cfg,
        predictions=predictions,
        labels=labels,
        n_permutations=50,
        threshold=0.5,
        output_dir=tmp_path,
        verbose=False,
    )
    assert results.metrics["f1"].observed == pytest.approx(1.0)

    name = experiment_name(cfg)
    files = list(tmp_path.iterdir())
    assert files
    for fp in files:
        assert name in fp.name, f"file {fp.name!r} missing experiment signature"


def test_run_inference_test_no_output_dir_skips_artifacts(cfg) -> None:
    rng = np.random.default_rng(1)
    n = 100
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    predictions = labels.copy().astype(np.float32)
    results = run_inference_test(
        cfg=cfg,
        predictions=predictions,
        labels=labels,
        n_permutations=20,
        verbose=False,
    )
    assert results.n_permutations == 20


def test_run_retrain_test_with_mock_trainer(tmp_path: Path, cfg) -> None:
    """End-to-end retrain run with an injected mock factory.

    Confirms that the runner doesn't depend on the Phase-2 stub when a
    ``trainer_factory`` is supplied explicitly.
    """

    class _MockTrainer:
        def reset_for_new_permutation(self) -> None:
            pass

        def train_and_evaluate(self) -> dict[str, float]:
            return {"f1": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5}

    factory_calls = {"n": 0}

    def factory(**_kwargs: object) -> _MockTrainer:
        factory_calls["n"] += 1
        return _MockTrainer()

    results = run_retrain_test(
        cfg=cfg,
        observed_metrics={"f1": 0.95, "accuracy": 0.9, "precision": 0.9, "recall": 0.9},
        trainer_factory=factory,
        n_permutations=3,
        permutation_epochs=1,
        output_dir=tmp_path,
        verbose=False,
    )
    assert factory_calls["n"] == 1  # legacy semantic: trainer constructed once
    assert results.metrics["f1"].observed == 0.95
    # Output dir got experiment-aware artifacts.
    name = experiment_name(cfg)
    files = list(tmp_path.iterdir())
    assert files
    for fp in files:
        assert name in fp.name
