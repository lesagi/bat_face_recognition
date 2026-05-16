from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bat_cli.__main__ import main
from bat_cli.runtime import (
    PermutationResult,
    _audit_params,
    _build_train_generator,
    _maybe_promote,
    _sample_explanation_records,
    _sample_projection_records,
    build_trainer_config,
    compose_config,
)
from bat_core import ImageRecord, Manifest
from click.testing import CliRunner


def test_help_lists_core_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "sweep" in result.output
    assert "build-manifest" in result.output
    assert "permutation-test" in result.output
    assert "compare-champion" in result.output


def test_compare_champion_help_lists_required_options() -> None:
    result = CliRunner().invoke(main, ["compare-champion", "--help"])

    assert result.exit_code == 0
    for flag in ("--model-name", "--output", "--tracking-uri"):
        assert flag in result.output


def test_compare_champion_command_invokes_runtime(tmp_path: Path) -> None:
    output = tmp_path / "champion.html"
    with patch("bat_reporting.compare_to_champion", return_value=output) as fake:
        result = CliRunner().invoke(
            main,
            [
                "compare-champion",
                "run-CANDIDATE",
                "--model-name",
                "arcface-model",
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["candidate_run_id"] == "run-CANDIDATE"
    assert kwargs["model_name"] == "arcface-model"
    assert kwargs["output_path"] == output
    assert "Champion comparison report" in result.output


def test_compare_champion_surfaces_no_champion_error(tmp_path: Path) -> None:
    """When the registry has no champion, the user sees a clear error."""
    from bat_reporting import NoChampionError

    with patch(
        "bat_reporting.compare_to_champion",
        side_effect=NoChampionError("no Production-stage version of 'arcface-model'"),
    ):
        result = CliRunner().invoke(
            main,
            [
                "compare-champion",
                "run-CANDIDATE",
                "--model-name",
                "arcface-model",
                "--output",
                str(tmp_path / "champion.html"),
            ],
        )

    assert result.exit_code != 0
    assert "no Production-stage version" in result.output


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


def test_build_trainer_config_maps_val_verification_override(tmp_path: Path) -> None:
    cfg = compose_config(
        experiment="arcface_rousettus_random_bg_video",
        overrides=["trainer.val_verification=false"],
    )

    trainer_cfg = build_trainer_config(cfg, output_dir=tmp_path / "run")

    assert trainer_cfg.val_verification is False


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


def test_train_help_advertises_promotion_and_explanation_flags() -> None:
    result = CliRunner().invoke(main, ["train", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--promote",
        "--prompt-promote",
        "--promotion-criterion",
        "--no-explanations",
        "--explanations-per-split",
    ):
        assert flag in result.output


def test_train_promote_and_prompt_promote_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(
        main,
        [
            "train",
            "--experiment",
            "arcface_rousettus_random_bg_video",
            "--no-mlflow",
            "--promote",
            "--prompt-promote",
            "--hydra",
            "data.manifest_path=/tmp/does-not-exist.csv",
        ],
    )

    assert result.exit_code != 0
    assert "--promote and --prompt-promote are mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# _maybe_promote: promotion gate logic
# ---------------------------------------------------------------------------


def _fake_tracker(beats: bool, candidate: float | None, incumbent: float | None) -> MagicMock:
    tracker = MagicMock()
    tracker.would_promote.return_value = (beats, candidate, incumbent)
    tracker.promote_to_champion.return_value = True
    return tracker


def test_maybe_promote_off_by_default() -> None:
    tracker = _fake_tracker(True, 0.95, 0.88)
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=False,
        prompt_promote=False,
        confirm=None,
        warnings=[],
    )

    assert decision.mode == "off"
    assert decision.promoted is False
    tracker.would_promote.assert_not_called()
    tracker.promote_to_champion.assert_not_called()


def test_maybe_promote_auto_silently_promotes_when_better() -> None:
    tracker = _fake_tracker(True, 0.95, 0.88)
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=True,
        prompt_promote=False,
        confirm=None,
        warnings=[],
    )

    assert decision.mode == "auto"
    assert decision.beats is True
    assert decision.promoted is True
    assert decision.declined is False
    tracker.promote_to_champion.assert_called_once_with("run-NEW", "test/roc_auc")


def test_maybe_promote_auto_skips_when_worse() -> None:
    tracker = _fake_tracker(False, 0.7, 0.92)
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=True,
        prompt_promote=False,
        confirm=None,
        warnings=[],
    )

    assert decision.mode == "auto"
    assert decision.beats is False
    assert decision.promoted is False
    tracker.promote_to_champion.assert_not_called()


def test_maybe_promote_prompt_skips_when_worse_without_asking() -> None:
    tracker = _fake_tracker(False, 0.7, 0.92)
    confirm = MagicMock()
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=False,
        prompt_promote=True,
        confirm=confirm,
        warnings=[],
    )

    assert decision.mode == "prompt"
    assert decision.beats is False
    assert decision.promoted is False
    confirm.assert_not_called()
    tracker.promote_to_champion.assert_not_called()


def test_maybe_promote_prompt_promotes_when_user_approves() -> None:
    tracker = _fake_tracker(True, 0.95, 0.88)
    confirm = MagicMock(return_value=True)
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=False,
        prompt_promote=True,
        confirm=confirm,
        warnings=[],
    )

    assert decision.mode == "prompt"
    assert decision.beats is True
    assert decision.promoted is True
    assert decision.declined is False
    confirm.assert_called_once_with(True, 0.95, 0.88)
    tracker.promote_to_champion.assert_called_once()


def test_maybe_promote_prompt_records_decline_when_user_says_no() -> None:
    tracker = _fake_tracker(True, 0.95, 0.88)
    confirm = MagicMock(return_value=False)
    decision = _maybe_promote(
        tracker=tracker,
        run_id="run-NEW",
        criterion="test/roc_auc",
        promote=False,
        prompt_promote=True,
        confirm=confirm,
        warnings=[],
    )

    assert decision.mode == "prompt"
    assert decision.beats is True
    assert decision.declined is True
    assert decision.promoted is False
    tracker.promote_to_champion.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-split explanation sampler
# ---------------------------------------------------------------------------


def _record_for(split: str, identity: str, idx: int = 0) -> ImageRecord:
    return ImageRecord(
        path=Path(f"data/{split}/{identity}_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,  # type: ignore[arg-type]
        quality=0.5,
    )


def test_sample_explanation_records_spans_all_splits() -> None:
    records = [
        _record_for("train", "A"),
        _record_for("train", "B"),
        _record_for("train", "C"),
        _record_for("val", "X"),
        _record_for("val", "Y"),
        _record_for("test", "P"),
        _record_for("test", "Q"),
    ]
    manifest = Manifest.from_records(records)
    samples, path_to_split = _sample_explanation_records(manifest, samples_per_split=2)

    splits = {path_to_split[s.path] for s in samples}
    assert splits == {"train", "val", "test"}
    train_samples = [s for s in samples if path_to_split[s.path] == "train"]
    assert len(train_samples) == 2  # two identities sampled from train
    assert {s.identity for s in train_samples} == {"A", "B"}  # alphabetical, deterministic


def test_sample_explanation_records_skips_empty_splits() -> None:
    records = [_record_for("train", "A"), _record_for("test", "Z")]
    manifest = Manifest.from_records(records)
    samples, path_to_split = _sample_explanation_records(manifest, samples_per_split=2)

    splits = {path_to_split[s.path] for s in samples}
    assert splits == {"train", "test"}  # val is silently skipped


def test_sample_projection_records_caps_at_limit() -> None:
    records = [_record_for("train", f"id{i:02d}") for i in range(60)]
    manifest = Manifest.from_records(records)
    sample = _sample_projection_records(manifest, cap=10)

    assert len(sample) == 10


def test_sample_projection_records_returns_all_when_under_cap() -> None:
    records = [_record_for("train", "A"), _record_for("val", "B")]
    manifest = Manifest.from_records(records)
    sample = _sample_projection_records(manifest, cap=30)

    assert len(sample) == 2


# ---------------------------------------------------------------------------
# Original suite
# ---------------------------------------------------------------------------


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


def test_build_explanations_routes_image_size_to_both_adapters(tmp_path: Path) -> None:
    """`_build_explanations` must override the adapters' default input_size.

    The `EmbeddingProjectionAdapter` dataclass defaults to ``input_size=224``,
    which silently mismatches a Siamese head trained at 105 and cascade-fails
    the whole explanation block. The CLI is responsible for forwarding the
    model's actual training edge length, sourced from
    `cfg.model.input_edge_length`.
    """
    from dataclasses import dataclass
    from unittest.mock import MagicMock

    from bat_cli.runtime import _build_explanations

    captured_inputs: dict[str, int] = {}

    @dataclass
    class _CaptureAdapter:
        input_size: int = -1

        def explain(self, _model, _samples):  # noqa: ANN001 - mock
            captured_inputs["adapter"] = self.input_size
            return []

    def fake_select_adapter(_model):  # noqa: ANN001 - mock
        return _CaptureAdapter()

    class _FakeProjection:
        def __init__(self, *, method: str, output_dir: Path, input_size: int) -> None:
            captured_inputs["projection"] = input_size

        def explain(self, _model, _samples):  # noqa: ANN001 - mock
            return []

    fake_model = MagicMock()
    fake_model.family = "pair"
    fake_manifest = MagicMock()

    # Force at least one sample so the function reaches the adapter calls.
    fake_record = MagicMock()
    fake_record.path = Path("/dev/null/fake.png")
    fake_record.identity = "X"
    fake_manifest.filter_split.side_effect = lambda split: (  # noqa: ARG005
        [fake_record] if split == "train" else []
    )

    with (
        patch("bat_interpretability.EmbeddingProjectionAdapter", new=_FakeProjection),
        patch("bat_interpretability.select_adapter", new=fake_select_adapter),
    ):
        _build_explanations(
            fake_model,
            fake_manifest,
            output_dir=tmp_path,
            samples_per_split=1,
            projection_cap=1,
            image_size=105,
        )

    assert captured_inputs["adapter"] == 105
    assert captured_inputs["projection"] == 105


def test_audit_params_emits_keys_that_survive_hp_filter() -> None:
    """`_audit_params` must produce keys the bat_tracking allowlist keeps.

    The MLflow run for the first PyTorch Siamese parity attempt logged zero
    params because the previous flattener emitted dotted Hydra keys
    (`trainer.lr`) that `bat_tracking.filter_params` drops. This pins the
    contract that the audit-translator's output round-trips through
    `filter_params` non-empty.
    """
    from bat_tracking.hp_audit import filter_params

    cfg = {
        "model": {
            "family": "pair",
            "arch": "siamese_4conv",
            "embedding_dim": 4096,
        },
        "loss": {"type": "binary_focal", "alpha": 0.75, "gamma": 2.0},
        "trainer": {
            "optimizer": "adam",
            "lr": 5.0e-5,
            "weight_decay": 0.0,
            "epochs": 80,
            "gradient_accumulation_steps": 1,
            "warmup_epochs": 0,
            "ema_decay": 0.0,
            "early_stop": {"patience": 5, "monitor": "val/f1"},
        },
        "data": {
            "species": "rousettus",
            "background": "random",
            "source": "video",
            "split_mode": "identity_disjoint",
            "split_seed": 7,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "batch_size": 32,
        },
    }

    audited = _audit_params(cfg, manifest_hash="deadbeef")
    kept = filter_params(audited)

    # Core knobs must survive the audit.
    for key in (
        "model_family",
        "model_arch",
        "embedding_dim",
        "loss_type",
        "optimizer",
        "lr",
        "weight_decay",
        "epochs",
        "early_stop_patience",
        "early_stop_monitor",
        "split_mode",
        "split_seed",
        "species",
        "background",
        "data_source",
        "manifest_hash",
    ):
        assert key in kept, f"audit dropped {key!r}; kept keys: {sorted(kept)}"

    # Loss sub-params are routed through the `loss_params.` prefix.
    assert kept["loss_params.alpha"] == 0.75
    assert kept["loss_params.gamma"] == 2.0

    # The previous flattener emitted these dotted forms; they must NOT appear.
    assert "trainer.lr" not in kept
    assert "model.family" not in kept

    # AdaFace's `h` knob (configs/loss/adaface.yaml) lives alongside
    # margin/scale and must also survive the audit.
    adaface_cfg = {
        "model": {"family": "embedding", "arch": "adaface", "embedding_dim": 512},
        "loss": {"type": "adaface", "margin": 0.4, "h": 0.333, "scale": 64.0},
        "trainer": {"optimizer": "adam", "lr": 1.0e-3, "epochs": 30},
        "data": {"species": "rousettus", "background": "random", "source": "video"},
    }
    adaface_kept = filter_params(_audit_params(adaface_cfg, manifest_hash="deadbeef"))
    assert adaface_kept["loss_params.margin"] == 0.4
    assert adaface_kept["loss_params.h"] == 0.333
    assert adaface_kept["loss_params.scale"] == 64.0


def test_build_train_generator_returns_none_when_deterministic_off() -> None:
    """No deterministic flag → no generator (preserves pre-fix DataLoader behavior)."""
    assert _build_train_generator({"trainer": {"deterministic": False}}) is None
    assert _build_train_generator({"trainer": {}}) is None
    assert _build_train_generator({}) is None


def test_build_train_generator_yields_deterministic_shuffle_order() -> None:
    """The end-to-end contract: identical cfg → identical batch order.

    Phase-4 follow-up #33 documented that two ArcFace runs with
    `trainer.deterministic=true` diverged from the first batch because
    `DataLoader(shuffle=True)` was constructed without an explicit
    generator. This pins the fix: two `_build_train_generator` calls
    with the same seed produce DataLoaders that iterate in lock-step.
    """
    pytest.importorskip("torch")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = {"trainer": {"deterministic": True}, "seed": 42}

    dataset = TensorDataset(torch.arange(20).unsqueeze(1))

    def _first_two_batches(gen) -> list[list[int]]:  # noqa: ANN001 - mock
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=gen)
        return [batch[0].squeeze(-1).tolist() for batch, in [next(iter(loader))]] + [
            batch[0].squeeze(-1).tolist() for batch, in [list(loader)[1]]
        ]

    order_a = _first_two_batches(_build_train_generator(cfg))
    order_b = _first_two_batches(_build_train_generator(cfg))

    assert order_a == order_b, (
        f"deterministic generator failed to produce stable batch order: " f"{order_a} vs {order_b}"
    )

    # And: omitting the deterministic flag must NOT return a generator
    # (so the existing non-deterministic call path is untouched).
    assert _build_train_generator({"trainer": {"deterministic": False}, "seed": 42}) is None


def test_manifest_pair_dataset_is_reproducible_and_label_balanced(tmp_path: Path) -> None:
    """ManifestPairDataset must yield bit-exact same pair for the same idx.

    Phase-4 Step-2 root cause was an O(n) deterministic walk that gave the
    model only ~972 unique pairs per epoch on the rousettus train split.
    The replacement uses a per-index `random.Random(seed + idx)` to produce
    O(n²)-style diversity while staying bit-exact for dataloader replay /
    checkpoint resume.
    """
    from bat_cli.runtime import ManifestPairDataset

    # Synthetic manifest: 3 identities × 4 records each, all in "train".
    records = [
        ImageRecord(
            path=tmp_path / f"{identity}_{i}.png",
            identity=identity,
            species="rousettus",
            background="random",
            source="video",
            split="train",
            quality=1.0,
        )
        for identity in ("A", "B", "C")
        for i in range(4)
    ]
    manifest = Manifest(records=records, manifest_hash="testhash")

    # Patch the image loader to avoid real I/O — return the path as a sentinel.
    def fake_loader(path: Path, image_size: int) -> str:  # noqa: ARG001 - mock
        return str(path.name)

    with patch("bat_data.default_image_loader", new=fake_loader):
        ds = ManifestPairDataset(manifest, "train", image_size=105, pairs_per_record=10, seed=42)
        # Same idx → same pair, twice.
        a1, b1, lab1 = ds[7]
        a2, b2, lab2 = ds[7]
        assert (a1, b1, float(lab1)) == (a2, b2, float(lab2))

        # Even idx is positive (label 1.0), odd idx is negative (label 0.0).
        for even in (0, 2, 100):
            assert float(ds[even][2]) == 1.0
        for odd in (1, 3, 101):
            assert float(ds[odd][2]) == 0.0

        # Pair diversity: across many positive draws, anchor should pair with
        # multiple distinct same-identity partners (the bug we fixed gave a
        # single fixed partner per anchor).
        partners_for_first_anchor: set[str] = set()
        for k in range(0, 200, 2):  # positives only
            anchor_name, partner_name, _ = ds[k]
            if anchor_name == "A_0.png":
                partners_for_first_anchor.add(partner_name)
        assert (
            len(partners_for_first_anchor) >= 2
        ), f"sampler regressed to single-partner mode: {partners_for_first_anchor}"
