"""compare_runs: side-by-side HTML report against mocked MLflow runs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("jinja2")

from bat_reporting.compare import (  # noqa: E402
    NoChampionError,
    RunComparisonInputs,
    compare_runs,
    compare_to_champion,
)


class _FakeMlflowClient:
    def __init__(self, runs: dict[str, dict[str, Any]]) -> None:
        self._runs = runs

    def get_run(self, run_id: str) -> SimpleNamespace:
        payload = self._runs[run_id]
        return SimpleNamespace(
            data=SimpleNamespace(
                metrics=payload.get("metrics", {}),
                tags=payload.get("tags", {}),
            )
        )


def test_compare_runs_renders_html_via_mocked_client(tmp_path: Path) -> None:
    fake = _FakeMlflowClient(
        {
            "run-A": {
                "metrics": {
                    "test/roc_auc": 0.84,
                    "test/top1": 0.71,
                    "val/f1": 0.77,
                },
                "tags": {"experiment_name": "rousettus_video_random_arcface_arcface"},
            },
            "run-B": {
                "metrics": {
                    "test/roc_auc": 0.88,
                    "test/top1": 0.74,
                    "val/f1": 0.79,
                },
                "tags": {"experiment_name": "rousettus_video_random_arcface_adaface"},
            },
        }
    )
    output = tmp_path / "compare.html"
    result = compare_runs("run-A", "run-B", output, mlflow_client=fake)

    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert text.strip(), "HTML output is empty"
    assert "run-A" in text and "run-B" in text
    assert "test/roc_auc" in text
    # Delta column: B - A = 0.04 -> formatted as 0.0400.
    assert "0.0400" in text


def test_compare_runs_prebuilt_inputs_with_overlays(tmp_path: Path) -> None:
    cfg = {
        "data": {"species": "mauritius", "source": "still", "background": "green"},
        "model": {"name": "siamese"},
        "loss": {"name": "bce"},
    }
    a = RunComparisonInputs(
        run_id="A",
        metrics={"test/roc_auc": 0.80, "test/top1": 0.65},
        cfg=cfg,
        roc_curve=([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.85, 1.0]),
        cmc_curve=[0.6, 0.75, 0.85],
    )
    b = RunComparisonInputs(
        run_id="B",
        metrics={"test/roc_auc": 0.83, "test/top1": 0.70},
        cfg=cfg,
        roc_curve=([0.0, 0.1, 0.5, 1.0], [0.0, 0.5, 0.9, 1.0]),
        cmc_curve=[0.65, 0.80, 0.90],
    )
    out = tmp_path / "compare_overlays.html"
    result = compare_runs("A", "B", out, inputs_a=a, inputs_b=b, plot_cfg=cfg)
    text = result.read_text(encoding="utf-8")
    assert "ROC overlay" in text
    assert "CMC overlay" in text

    # The overlay PNGs landed alongside the HTML and follow naming convention.
    pngs = list(tmp_path.glob("*.png"))
    assert any("roc_overlay" in p.name for p in pngs)
    assert any("cmc_overlay" in p.name for p in pngs)
    for png in pngs:
        assert "mauritius" in png.name and "siamese" in png.name


# ---------------------------------------------------------------------------
# compare_to_champion
# ---------------------------------------------------------------------------


def test_compare_to_champion_renders_with_champion_on_left(tmp_path: Path) -> None:
    """Champion run_id resolved via get_champion lands on the left (run_a)."""
    fake_client = _FakeMlflowClient(
        {
            "run-CHAMPION": {
                "metrics": {"test/roc_auc": 0.84, "val/f1": 0.77},
                "tags": {"experiment_name": "rousettus_video_random_arcface_arcface"},
            },
            "run-CANDIDATE": {
                "metrics": {"test/roc_auc": 0.91, "val/f1": 0.85},
                "tags": {"experiment_name": "rousettus_video_random_arcface_adaface"},
            },
        }
    )
    fake_champion = SimpleNamespace(run_id="run-CHAMPION", version="3", name="arcface-model")

    output = tmp_path / "champion.html"
    with patch("bat_tracking.get_champion", return_value=fake_champion) as fake_gc:
        result = compare_to_champion(
            candidate_run_id="run-CANDIDATE",
            model_name="arcface-model",
            output_path=output,
            mlflow_client=fake_client,
        )

    fake_gc.assert_called_once_with("arcface-model", client=fake_client)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    # Champion is run_a (left), candidate is run_b (right).
    assert "run-CHAMPION" in text and "run-CANDIDATE" in text
    # Delta column: candidate - champion = 0.91 - 0.84 = 0.07.
    assert "0.0700" in text


def test_compare_to_champion_raises_when_no_champion(tmp_path: Path) -> None:
    """No Production-stage version -> NoChampionError with a useful message."""
    fake_client = _FakeMlflowClient({"run-CANDIDATE": {"metrics": {}}})
    with (
        patch("bat_tracking.get_champion", return_value=None),
        pytest.raises(NoChampionError, match="no Production-stage version"),
    ):
        compare_to_champion(
            candidate_run_id="run-CANDIDATE",
            model_name="arcface-model",
            output_path=tmp_path / "champion.html",
            mlflow_client=fake_client,
        )


def test_compare_to_champion_accepts_prebuilt_inputs(tmp_path: Path) -> None:
    """Prebuilt RunComparisonInputs bypass MLflow but champion resolution still happens."""
    fake_champion = SimpleNamespace(run_id="run-CHAMPION", version="1", name="m")
    champion_inputs = RunComparisonInputs(run_id="run-CHAMPION", metrics={"test/roc_auc": 0.5})
    candidate_inputs = RunComparisonInputs(run_id="run-CANDIDATE", metrics={"test/roc_auc": 0.9})

    output = tmp_path / "champion.html"
    with patch("bat_tracking.get_champion", return_value=fake_champion):
        result = compare_to_champion(
            candidate_run_id="run-CANDIDATE",
            model_name="m",
            output_path=output,
            inputs_champion=champion_inputs,
            inputs_candidate=candidate_inputs,
        )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "run-CHAMPION" in text and "run-CANDIDATE" in text
    # 0.9 - 0.5 = 0.4
    assert "0.4000" in text
