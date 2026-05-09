from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from bat_cli.__main__ import main
from bat_cli.runtime import PermutationResult, compose_config
from click.testing import CliRunner


def test_help_lists_core_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "sweep" in result.output
    assert "build-manifest" in result.output
    assert "permutation-test" in result.output


def test_train_help_advertises_permutation_flags() -> None:
    result = CliRunner().invoke(main, ["train", "--help"])

    assert result.exit_code == 0
    assert "--no-permutation" in result.output
    assert "--permutation-n" in result.output


def test_permutation_test_help_lists_required_options() -> None:
    result = CliRunner().invoke(main, ["permutation-test", "--help"])

    assert result.exit_code == 0
    for flag in ("--experiment", "--n-permutations", "--seed", "--threshold", "--degradation"):
        assert flag in result.output


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


def test_permutation_test_command_invokes_runtime(tmp_path: Path) -> None:
    fake_result = PermutationResult(
        output_dir=tmp_path / "perm",
        n_permutations=42,
        metrics_tested=("f1", "accuracy"),
        p_values={"f1": 0.01, "accuracy": 0.04},
    )
    with patch("bat_cli.__main__.run_permutation_test", return_value=fake_result) as fake:
        result = CliRunner().invoke(
            main,
            [
                "permutation-test",
                "--experiment",
                "arcface_rousettus_random_bg_video",
                "--n-permutations",
                "42",
                "--seed",
                "7",
                "--hydra",
                "data.manifest_path=/tmp/does-not-exist.csv",
            ],
        )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["n_permutations"] == 42
    assert kwargs["seed"] == 7
    assert kwargs["degradation"] is False
    assert "p_values" in result.output
    assert "f1" in result.output


def test_permutation_test_rejects_malformed_hydra_override() -> None:
    result = CliRunner().invoke(
        main,
        [
            "permutation-test",
            "--experiment",
            "arcface_rousettus_random_bg_video",
            "--hydra",
            "no-equals-sign",
        ],
    )

    assert result.exit_code != 0
    assert "Hydra override" in result.output


def test_train_dry_run_exposes_no_permutation_flag() -> None:
    result = CliRunner().invoke(
        main,
        [
            "train",
            "--experiment",
            "arcface_rousettus_random_bg_video",
            "--dry-run",
            "--no-mlflow",
            "--no-permutation",
            "--hydra",
            "data.manifest_path=/tmp/does-not-exist.csv",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run OK" in result.output


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
