from __future__ import annotations

from pathlib import Path

import pytest
from bat_cli.__main__ import main
from bat_cli.runtime import compose_config
from click.testing import CliRunner


def test_help_lists_core_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "sweep" in result.output
    assert "build-manifest" in result.output


def test_compose_arcface_experiment_uses_global_overrides() -> None:
    cfg = compose_config(experiment="arcface_rousettus_random_bg_video")

    assert cfg["model"]["head"] == "arcface"
    assert cfg["loss"]["type"] == "arcface"
    assert cfg["trainer"]["family"] == "embedding"
    assert cfg["data"]["background"] == "random"


def test_train_dry_run_does_not_require_manifest() -> None:
    result = CliRunner().invoke(
        main,
        [
            "train",
            "--experiment",
            "arcface_rousettus_random_bg_video",
            "--dry-run",
            "--no-mlflow",
            "--hydra",
            "data.manifest_path=/tmp/does-not-exist.csv",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run OK" in result.output
    assert "arcface" in result.output


def test_build_manifest_command_writes_split_csv(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")

    for identity in ("A", "B", "C"):
        image_path = tmp_path / f"r--{identity}--000.png"
        pil.new("RGB", (8, 8), color=(120, 80, 40)).save(image_path)

    output_csv = tmp_path / "manifest.csv"
    result = CliRunner().invoke(
        main,
        [
            "build-manifest",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output_csv),
            "--species",
            "rousettus",
            "--val-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
        ],
    )

    assert result.exit_code == 0
    assert output_csv.exists()
    assert output_csv.with_suffix(".csv.hash").exists()
