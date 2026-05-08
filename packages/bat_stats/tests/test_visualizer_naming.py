"""Naming regression test: every filename emitted by ``visualizer.py``
must include the species/source/background/model/loss components.

This is the single most important test in the bat_stats package -- the user
explicitly flagged the legacy naming gap as a methodology bug, and this
guards against any plot escaping with a generic name.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402

from bat_stats.naming import experiment_name  # noqa: E402
from bat_stats.permutation_test import (  # noqa: E402
    MetricResult,
    PermutationTestResults,
)
from bat_stats.visualizer import PermutationVisualizer  # noqa: E402

EXPECTED_TOKENS = ("rousettus", "video", "random", "arcface")


def _build_results() -> PermutationTestResults:
    rng = np.random.default_rng(7)
    return PermutationTestResults(
        n_permutations=50,
        significance_level=0.05,
        permutation_epochs=0,
        metrics={
            "f1": MetricResult(
                observed=0.92,
                null_mean=0.50,
                null_std=0.05,
                p_value=0.01,
                significant=True,
                null_distribution=rng.normal(0.5, 0.05, size=50).tolist(),
            ),
            "accuracy": MetricResult(
                observed=0.91,
                null_mean=0.50,
                null_std=0.04,
                p_value=0.02,
                significant=True,
                null_distribution=rng.normal(0.5, 0.04, size=50).tolist(),
            ),
        },
        total_time_seconds=0.5,
    )


def _assert_experiment_aware(filename: str, cfg: dict) -> None:
    """Every component of the experiment signature must be in the filename."""

    name = experiment_name(cfg)
    assert name in filename, (
        f"filename {filename!r} is missing the full experiment signature {name!r}"
    )
    for token in EXPECTED_TOKENS:
        assert token in filename, (
            f"filename {filename!r} missing required token {token!r}"
        )


def test_individual_filenames_are_experiment_aware(tmp_path: Path, cfg) -> None:
    results = _build_results()
    viz = PermutationVisualizer(results, cfg=cfg, output_dir=tmp_path)

    viz.plot_null_distribution("f1", show=False, save=True)
    viz.plot_null_distribution("accuracy", show=False, save=True)
    viz.plot_all_distributions(show=False, save=True)
    viz.plot_summary_table(show=False, save=True)
    viz.plot_degradation_curve(
        {"f1": [(0.0, 0.92), (0.5, 0.7), (1.0, 0.5)]},
        show=False,
        save=True,
    )
    viz.export_csv()
    viz.export_json()

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files, "no output files were produced"
    for fname in files:
        _assert_experiment_aware(fname, cfg)


def test_full_report_filenames_are_experiment_aware(tmp_path: Path, cfg) -> None:
    """``generate_full_report`` is the all-in-one path used by the runner."""

    results = _build_results()
    viz = PermutationVisualizer(results, cfg=cfg, output_dir=tmp_path)
    artifacts = viz.generate_full_report(
        show=False,
        export_csv=True,
        export_json=True,
        degradation_data={"f1": [(0.0, 0.92), (0.5, 0.71), (1.0, 0.50)]},
    )
    assert artifacts, "expected at least one artifact"
    for path in artifacts:
        _assert_experiment_aware(path.name, cfg)


def test_no_generic_names_leak(tmp_path: Path, cfg) -> None:
    """Sanity: legacy generic names like ``null_dist_f1.png`` never appear."""

    results = _build_results()
    viz = PermutationVisualizer(results, cfg=cfg, output_dir=tmp_path)
    viz.generate_full_report(
        show=False,
        export_csv=True,
        export_json=True,
        degradation_data={"accuracy": [(0.0, 0.91), (1.0, 0.50)]},
    )
    forbidden = {
        "null_dist_f1.png",
        "null_dist_accuracy.png",
        "null_distributions_all.png",
        "permutation_summary.png",
        "permutation_results.csv",
        "permutation_results.json",
        "degradation_curve.png",
    }
    actual = {p.name for p in tmp_path.iterdir()}
    leaked = forbidden & actual
    assert not leaked, (
        f"generic filename(s) leaked: {leaked}; actual files={actual}"
    )


def test_filenames_change_with_cfg(tmp_path: Path) -> None:
    """A different cfg → different filenames (proves the embedding is real)."""

    cfg_a = {
        "species": "rousettus",
        "source": "video",
        "background": "random",
        "model": {"name": "arcface"},
        "loss": {"name": "arcface"},
    }
    cfg_b = {
        "species": "mauritius",
        "source": "still",
        "background": "green",
        "model": {"name": "siamese"},
        "loss": {"name": "bce"},
    }
    results = _build_results()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    PermutationVisualizer(results, cfg=cfg_a, output_dir=dir_a).plot_summary_table(
        show=False, save=True
    )
    PermutationVisualizer(results, cfg=cfg_b, output_dir=dir_b).plot_summary_table(
        show=False, save=True
    )

    files_a = {p.name for p in dir_a.iterdir()}
    files_b = {p.name for p in dir_b.iterdir()}
    assert files_a != files_b
    assert any("rousettus" in f for f in files_a)
    assert any("mauritius" in f for f in files_b)


def test_csv_includes_experiment_column(tmp_path: Path, cfg) -> None:
    """The CSV row format includes a per-row experiment-name token."""

    results = _build_results()
    viz = PermutationVisualizer(results, cfg=cfg, output_dir=tmp_path)
    csv_path = viz.export_csv()
    text = csv_path.read_text()
    assert "Experiment" in text.splitlines()[0]
    assert "rousettus_video_random_arcface_arcface" in text


def test_json_filename_is_experiment_aware(tmp_path: Path, cfg) -> None:
    results = _build_results()
    viz = PermutationVisualizer(results, cfg=cfg, output_dir=tmp_path)
    json_path = viz.export_json()
    _assert_experiment_aware(json_path.name, cfg)
    payload = json.loads(json_path.read_text())
    assert "metrics" in payload
