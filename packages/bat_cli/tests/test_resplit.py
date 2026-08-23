"""Runtime re-split: opt-in behaviour, seed derivation, and MLflow provenance.

The manifests on disk carry a baked ``split`` column and ``data.split_seed`` used
to be logged but never read, so every run of an experiment held out the same
bats. These tests pin down the replacement:

* it stays **off** unless asked for, so the 90 published runs are unaffected;
* ``--fold N`` derives the seed deterministically;
* ``manifest_hash`` keeps meaning the *on-disk* hash even though re-splitting
  rehashes the manifest, otherwise new runs stop being comparable to old ones;
* every provenance param survives the HP-audit allowlist — a silently dropped
  param would make a fold unreproducible from MLflow alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from bat_cli.__main__ import _apply_fold_overrides
from bat_cli.runtime import (
    CliRuntimeError,
    SplitInfo,
    TrainingBundle,
    _audit_params,
    _log_split_assignment,
    _split_tags,
    resplit_manifest,
)
from bat_core import ImageRecord, Manifest
from bat_tracking import filter_params

N_IDENTITIES = 16
PER_IDENTITY = 5


def _manifest(n_identities: int = N_IDENTITIES, per_identity: int = PER_IDENTITY) -> Manifest:
    records = []
    idx = 0
    for i in range(n_identities):
        for _ in range(per_identity):
            records.append(
                ImageRecord(
                    path=Path(f"/data/img_{idx}.png"),
                    identity=f"bat_{i:03d}",
                    species="mauritius",
                    background="green",
                    source="video",
                    augmented=False,
                    split="train",
                    quality=12.0,
                )
            )
            idx += 1
    return Manifest.from_records(records)


def _cfg(**data: object) -> dict[str, object]:
    base = {
        "manifest_path": "data/manifests/mauritius_green_aligned_224_manifest.csv",
        "species": "mauritius",
        "background": "green",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "split_seed": 42,
    }
    base.update(data)
    return {"data": base}


# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------


def test_resplit_is_off_by_default() -> None:
    manifest = _manifest()
    out, info = resplit_manifest(manifest, _cfg())
    assert info is None
    assert out is manifest


def test_resplit_off_leaves_the_baked_split_untouched() -> None:
    manifest = _manifest()
    out, _ = resplit_manifest(manifest, _cfg(resplit=False))
    assert out.identities("train") == manifest.identities("train")
    assert out.manifest_hash == manifest.manifest_hash


def test_resplit_true_repartitions() -> None:
    manifest = _manifest()
    out, info = resplit_manifest(manifest, _cfg(resplit=True))
    assert info is not None
    assert info.resplit is True
    out.assert_identity_disjoint()
    # The baked manifest was all-train; a re-split must populate val and test.
    assert out.identities("val")
    assert out.identities("test")


def test_fold_id_alone_enables_resplit() -> None:
    """Setting a fold implies re-splitting — no need to also pass resplit."""
    _, info = resplit_manifest(_manifest(), _cfg(fold_id=7))
    assert info is not None
    assert info.fold_id == 7


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def test_fold_id_derives_the_split_seed_from_the_base() -> None:
    _, info = resplit_manifest(_manifest(), _cfg(fold_id=5, fold_base_seed=1000))
    assert info is not None
    assert info.split_seed == 1005


def test_fold_base_seed_is_honoured() -> None:
    _, info = resplit_manifest(_manifest(), _cfg(fold_id=5, fold_base_seed=7000))
    assert info is not None
    assert info.split_seed == 7005


def test_without_fold_id_the_split_seed_comes_from_config() -> None:
    _, info = resplit_manifest(_manifest(), _cfg(resplit=True, split_seed=99))
    assert info is not None
    assert info.split_seed == 99
    assert info.fold_id is None


def test_same_fold_reproduces_the_same_assignment() -> None:
    cfg = _cfg(fold_id=3, split_size_mode="seeded", min_test_identities=2, max_test_identities=6)
    first_manifest, first = resplit_manifest(_manifest(), cfg)
    second_manifest, second = resplit_manifest(_manifest(), cfg)
    assert first is not None and second is not None
    assert first.assignment_sha256 == second.assignment_sha256
    assert first_manifest.manifest_hash == second_manifest.manifest_hash


def test_different_folds_give_different_assignments() -> None:
    shas = set()
    for fold in range(20):
        _, info = resplit_manifest(
            _manifest(),
            _cfg(
                fold_id=fold,
                split_size_mode="seeded",
                min_test_identities=2,
                max_test_identities=6,
                min_val_identities=2,
                max_val_identities=4,
                min_train_identities=6,
            ),
        )
        assert info is not None
        shas.add(info.assignment_sha256)
    assert len(shas) == 20


def test_seeded_mode_varies_the_held_out_identity_count() -> None:
    counts = set()
    for fold in range(20):
        _, info = resplit_manifest(
            _manifest(),
            _cfg(
                fold_id=fold,
                split_size_mode="seeded",
                min_test_identities=2,
                max_test_identities=6,
                min_val_identities=2,
                max_val_identities=4,
                min_train_identities=6,
            ),
        )
        assert info is not None
        counts.add(len(info.test_identities))
    assert counts == {2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Hash provenance
# ---------------------------------------------------------------------------


def test_source_manifest_hash_is_the_pre_split_hash() -> None:
    manifest = _manifest()
    out, info = resplit_manifest(manifest, _cfg(resplit=True))
    assert info is not None
    assert info.source_manifest_hash == manifest.manifest_hash
    # Manifest.from_records folds `split` into the digest, so the re-split
    # manifest must hash differently — that is exactly why both are recorded.
    assert out.manifest_hash != info.source_manifest_hash


def test_bundle_reports_the_source_hash_when_resplit(tmp_path: Path) -> None:
    manifest = _manifest()
    out, info = resplit_manifest(manifest, _cfg(resplit=True))
    bundle = TrainingBundle(
        cfg={},
        manifest=out,
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
        split_info=info,
    )
    assert bundle.source_manifest_hash == manifest.manifest_hash


def test_bundle_falls_back_to_manifest_hash_without_resplit() -> None:
    manifest = _manifest()
    bundle = TrainingBundle(
        cfg={},
        manifest=manifest,
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
    )
    assert bundle.source_manifest_hash == manifest.manifest_hash


# ---------------------------------------------------------------------------
# MLflow provenance
# ---------------------------------------------------------------------------


def _info() -> SplitInfo:
    _, info = resplit_manifest(
        _manifest(),
        _cfg(
            fold_id=4,
            split_size_mode="seeded",
            min_test_identities=2,
            max_test_identities=6,
            min_val_identities=2,
            max_val_identities=4,
            min_train_identities=6,
        ),
    )
    assert info is not None
    return info


def test_every_param_survives_the_hp_audit_allowlist() -> None:
    """A dropped param means the fold cannot be rebuilt from MLflow."""
    params = _info().to_params()
    kept = filter_params(dict(params))
    assert set(kept) == set(params), f"dropped: {set(params) - set(kept)}"


def test_params_record_what_is_needed_to_reproduce_the_split() -> None:
    params = _info().to_params()
    for required in (
        "split.resplit",
        "split.size_mode",
        "split.split_seed",
        "split.fold_id",
        "split.assignment_sha256",
        "split.source_manifest_hash",
        "split.n_train_ids",
        "split.n_val_ids",
        "split.n_test_ids",
        "split.n_train_imgs",
        "split.n_val_imgs",
        "split.n_test_imgs",
    ):
        assert required in params, required


def test_params_include_the_configured_bounds() -> None:
    params = _info().to_params()
    assert params["split.min_test_identities"] == 2
    assert params["split.max_test_identities"] == 6


def test_identity_and_image_counts_agree_with_the_manifest() -> None:
    info = _info()
    assert (
        len(info.train_identities) + len(info.val_identities) + len(info.test_identities)
        == N_IDENTITIES
    )
    assert (
        info.n_train_images + info.n_val_images + info.n_test_images == N_IDENTITIES * PER_IDENTITY
    )


def test_audit_params_logs_the_effective_split_seed() -> None:
    """``split_seed`` must be the seed that produced the partition.

    The config's ``data.split_seed`` is 42 for every experiment; a fold derives
    ``fold_base_seed + fold_id`` instead. Logging 42 would hand a reader the one
    number they need to reproduce the split, wrong.
    """
    info = _info()  # fold 4 -> 1004
    params = _audit_params(_cfg(), manifest_hash="abc", split_info=info)
    assert params["split_seed"] == 1004


def test_audit_params_uses_the_config_seed_without_resplit() -> None:
    params = _audit_params(_cfg(), manifest_hash="abc")
    assert params["split_seed"] == 42


def test_audit_params_and_split_params_do_not_collide() -> None:
    """MLflow params are immutable — the same key must not be logged twice.

    Both dicts go to ``log_params`` in the same run, so an overlapping key with
    a different value would raise mid-training.
    """
    info = _info()
    audit = _audit_params(_cfg(), manifest_hash="abc", split_info=info)
    overlap = set(audit) & set(info.to_params())
    assert not overlap, f"colliding param keys: {overlap}"


def test_split_tags_expose_fold_and_test_size() -> None:
    info = _info()
    bundle = TrainingBundle(
        cfg={},
        manifest=_manifest(),
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
        split_info=info,
    )
    tags = _split_tags(bundle)
    assert tags["fold"] == "4"
    assert tags["n_test_ids"] == str(len(info.test_identities))
    assert tags["split_size_mode"] == "seeded"


def test_split_tags_empty_without_resplit() -> None:
    bundle = TrainingBundle(
        cfg={},
        manifest=_manifest(),
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
    )
    assert _split_tags(bundle) == {}


def test_assignment_artifact_is_written_and_logged(tmp_path: Path) -> None:
    info = _info()
    bundle = TrainingBundle(
        cfg={},
        manifest=_manifest(),
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
        split_info=info,
    )
    tracker = MagicMock()
    _log_split_assignment(tracker, bundle, tmp_path)

    path = tmp_path / "split_assignment.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["fold_id"] == 4
    assert payload["assignment_sha256"] == info.assignment_sha256
    assert sorted(payload["identities"]) == ["test", "train", "val"]
    assert payload["identities"]["test"] == list(info.test_identities)
    tracker.log_params.assert_called_once()
    tracker.log_artifact.assert_called_once_with(path)


def test_assignment_artifact_skipped_without_resplit(tmp_path: Path) -> None:
    bundle = TrainingBundle(
        cfg={},
        manifest=_manifest(),
        model=None,
        loss=None,
        trainer_cfg=MagicMock(),
        train_loader=None,
        val_loader=None,
        test_loader=None,
        trainer_kwargs={},
    )
    tracker = MagicMock()
    _log_split_assignment(tracker, bundle, tmp_path)
    assert not (tmp_path / "split_assignment.json").exists()
    tracker.log_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_infeasible_design_raises_a_cli_error() -> None:
    """A too-large design must surface as a config error, not a bat_data traceback."""
    with pytest.raises(CliRuntimeError, match="runtime re-split failed"):
        resplit_manifest(
            _manifest(n_identities=6),
            _cfg(
                resplit=True,
                split_size_mode="seeded",
                min_test_identities=3,
                min_val_identities=3,
                min_train_identities=3,
            ),
        )


def test_invalid_bounds_raise_a_cli_error() -> None:
    with pytest.raises(CliRuntimeError, match="runtime re-split failed"):
        resplit_manifest(
            _manifest(),
            _cfg(resplit=True, min_test_identities=6, max_test_identities=2),
        )


# ---------------------------------------------------------------------------
# --fold override expansion
# ---------------------------------------------------------------------------


def test_fold_flag_expands_to_the_expected_overrides() -> None:
    out = _apply_fold_overrides((), 3)
    assert "data.resplit=true" in out
    assert "data.fold_id=3" in out
    assert "data.split_size_mode=seeded" in out
    assert "seed=3" in out
    assert "trainer.deterministic=true" in out


def test_fold_flag_is_a_no_op_when_absent() -> None:
    assert _apply_fold_overrides(("trainer.epochs=5",), None) == ("trainer.epochs=5",)


def test_explicit_override_wins_over_the_fold_default() -> None:
    """A caller pinning the training seed must keep it while the split varies."""
    out = _apply_fold_overrides(("seed=42",), 3)
    assert "seed=42" in out
    assert "seed=3" not in out
    assert "data.fold_id=3" in out


def test_negative_fold_rejected() -> None:
    with pytest.raises(click.BadParameter):
        _apply_fold_overrides((), -1)


# ---------------------------------------------------------------------------
# Singleton-final-batch guard
# ---------------------------------------------------------------------------


class _CountedDataset:
    """Minimal sized dataset for exercising _loader's drop_last logic."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def test_loader_drops_a_lone_trailing_sample() -> None:
    """A final batch of 1 crashes BatchNorm, so it must be dropped.

    Surfaced by the seeded k-fold: varying fold sizes eventually produce a train
    count where ``len % batch_size == 1``, which the fixed baked splits never did.
    """
    from bat_cli.runtime import _loader

    loader = _loader(_CountedDataset(33), batch_size=32, shuffle=True, num_workers=0)
    assert loader.drop_last is True
    assert len(list(loader)) == 1  # the lone 33rd sample is dropped


def test_loader_keeps_trailing_batches_larger_than_one() -> None:
    """Dropping unconditionally would change training for every other run."""
    from bat_cli.runtime import _loader

    loader = _loader(_CountedDataset(34), batch_size=32, shuffle=True, num_workers=0)
    assert loader.drop_last is False
    assert len(list(loader)) == 2  # 32 + 2, nothing discarded


def test_loader_exact_multiple_keeps_all_batches() -> None:
    from bat_cli.runtime import _loader

    loader = _loader(_CountedDataset(64), batch_size=32, shuffle=True, num_workers=0)
    assert loader.drop_last is False
    assert len(list(loader)) == 2


def test_loader_never_drops_for_eval_loaders() -> None:
    """Eval must score every sample, so shuffle=False never drops."""
    from bat_cli.runtime import _loader

    loader = _loader(_CountedDataset(33), batch_size=32, shuffle=False, num_workers=0)
    assert loader.drop_last is False
    assert len(list(loader)) == 2
