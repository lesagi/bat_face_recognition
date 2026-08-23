"""Seeded split sizing: fractions as floors, seed decides the realised sizes.

Two things must hold simultaneously:

* ``size_mode="exact"`` is **byte-identical** to the behaviour that produced the
  90 already-published runs. That is what the frozen-expectation tests here
  pin down.
* ``size_mode="seeded"`` genuinely varies the number of held-out identities
  across seeds — otherwise the split-size sensitivity study has no independent
  variable — while never violating a floor, a cap, or identity-disjointness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bat_core import ImageRecord, Manifest  # noqa: E402
from bat_core.exceptions import InvalidManifestError  # noqa: E402
from bat_data.splitter import IdentitySplitter  # noqa: E402

# The two real datasets, so the tests exercise the sizes actually in use.
MAURITIUS_IDENTITIES = 16
ROUSETTUS_IDENTITIES = 12

# Locked ranges from the study design (docs plan, workstream B).
MAURITIUS_BOUNDS = {
    "min_test_identities": 2,
    "max_test_identities": 6,
    "min_val_identities": 2,
    "max_val_identities": 4,
    "min_train_identities": 6,
}
ROUSETTUS_BOUNDS = {
    "min_test_identities": 2,
    "max_test_identities": 5,
    "min_val_identities": 2,
    "max_val_identities": 3,
    "min_train_identities": 4,
}

FOLD_SEEDS = tuple(range(1000, 1020))  # the 20 fold seeds


def _make_record(idx: int, identity: str) -> ImageRecord:
    return ImageRecord(
        path=Path(f"/data/img_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split="train",
        quality=10.0 + idx,
    )


def _make_manifest(num_identities: int, per_id: int = 4) -> Manifest:
    records = []
    n = 0
    for i in range(num_identities):
        for _ in range(per_id):
            records.append(_make_record(n, f"bat_{i:03d}"))
            n += 1
    return Manifest.from_records(records)


def _identity_counts(manifest: Manifest) -> tuple[int, int, int]:
    return (
        len(manifest.identities("train")),
        len(manifest.identities("val")),
        len(manifest.identities("test")),
    )


# ---------------------------------------------------------------------------
# exact mode must not have moved
# ---------------------------------------------------------------------------


def test_exact_mode_is_the_default() -> None:
    assert IdentitySplitter().size_mode == "exact"


def test_exact_mode_reproduces_the_published_mauritius_split() -> None:
    """16 identities at 0.15/0.15 must still give 12 train / 2 val / 2 test."""
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    out = IdentitySplitter(val_fraction=0.15, test_fraction=0.15, seed=42).split(manifest)
    assert _identity_counts(out) == (12, 2, 2)


def test_exact_mode_reproduces_the_published_rousettus_split() -> None:
    """12 identities at 0.25/0.25 must still give 6 train / 3 val / 3 test."""
    manifest = _make_manifest(ROUSETTUS_IDENTITIES)
    out = IdentitySplitter(val_fraction=0.25, test_fraction=0.25, seed=42).split(manifest)
    assert _identity_counts(out) == (6, 3, 3)


def test_exact_mode_membership_is_unchanged_by_the_new_parameters() -> None:
    """Passing seeded-mode bounds must not perturb an exact-mode split."""
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    plain = IdentitySplitter(val_fraction=0.15, test_fraction=0.15, seed=42).split(manifest)
    with_bounds = IdentitySplitter(
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        size_mode="exact",
        **MAURITIUS_BOUNDS,
    ).split(manifest)
    for split in ("train", "val", "test"):
        assert plain.identities(split) == with_bounds.identities(split)


def test_exact_mode_sizes_are_constant_across_seeds() -> None:
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    sizes = {
        _identity_counts(
            IdentitySplitter(val_fraction=0.15, test_fraction=0.15, seed=seed).split(manifest)
        )
        for seed in FOLD_SEEDS
    }
    assert sizes == {(12, 2, 2)}


# ---------------------------------------------------------------------------
# seeded mode: sizes vary, bounds hold
# ---------------------------------------------------------------------------


def _seeded_splitter(bounds: dict[str, int], seed: int) -> IdentitySplitter:
    return IdentitySplitter(
        val_fraction=0.15,
        test_fraction=0.15,
        seed=seed,
        size_mode="seeded",
        **bounds,
    )


@pytest.mark.parametrize(
    ("n_identities", "bounds"),
    [
        (MAURITIUS_IDENTITIES, MAURITIUS_BOUNDS),
        (ROUSETTUS_IDENTITIES, ROUSETTUS_BOUNDS),
    ],
    ids=["mauritius", "rousettus"],
)
def test_seeded_mode_varies_the_test_size(n_identities: int, bounds: dict[str, int]) -> None:
    manifest = _make_manifest(n_identities)
    test_sizes = {
        _identity_counts(_seeded_splitter(bounds, seed).split(manifest))[2] for seed in FOLD_SEEDS
    }
    # The whole point of the mode: more than one held-out size across 20 folds.
    assert len(test_sizes) > 1


@pytest.mark.parametrize(
    ("n_identities", "bounds"),
    [
        (MAURITIUS_IDENTITIES, MAURITIUS_BOUNDS),
        (ROUSETTUS_IDENTITIES, ROUSETTUS_BOUNDS),
    ],
    ids=["mauritius", "rousettus"],
)
def test_seeded_mode_covers_the_whole_configured_range(
    n_identities: int, bounds: dict[str, int]
) -> None:
    """20 folds must exercise every test size in the declared range.

    If a level is never drawn, the sensitivity analysis has no data at it.
    """
    manifest = _make_manifest(n_identities)
    test_sizes = {
        _identity_counts(_seeded_splitter(bounds, seed).split(manifest))[2] for seed in FOLD_SEEDS
    }
    expected = set(range(bounds["min_test_identities"], bounds["max_test_identities"] + 1))
    assert test_sizes == expected


@pytest.mark.parametrize(
    ("n_identities", "bounds"),
    [
        (MAURITIUS_IDENTITIES, MAURITIUS_BOUNDS),
        (ROUSETTUS_IDENTITIES, ROUSETTUS_BOUNDS),
    ],
    ids=["mauritius", "rousettus"],
)
def test_seeded_mode_respects_every_bound(n_identities: int, bounds: dict[str, int]) -> None:
    manifest = _make_manifest(n_identities)
    for seed in FOLD_SEEDS:
        out = _seeded_splitter(bounds, seed).split(manifest)
        n_train, n_val, n_test = _identity_counts(out)
        assert bounds["min_test_identities"] <= n_test <= bounds["max_test_identities"]
        assert bounds["min_val_identities"] <= n_val <= bounds["max_val_identities"]
        assert n_train >= bounds["min_train_identities"]
        assert n_train + n_val + n_test == n_identities


@pytest.mark.parametrize(
    ("n_identities", "bounds"),
    [
        (MAURITIUS_IDENTITIES, MAURITIUS_BOUNDS),
        (ROUSETTUS_IDENTITIES, ROUSETTUS_BOUNDS),
    ],
    ids=["mauritius", "rousettus"],
)
def test_seeded_mode_keeps_splits_identity_disjoint(
    n_identities: int, bounds: dict[str, int]
) -> None:
    manifest = _make_manifest(n_identities)
    for seed in FOLD_SEEDS:
        out = _seeded_splitter(bounds, seed).split(manifest)
        out.assert_identity_disjoint()


def test_seeded_mode_is_reproducible_for_a_seed() -> None:
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    first = _seeded_splitter(MAURITIUS_BOUNDS, 1007).split(manifest)
    second = _seeded_splitter(MAURITIUS_BOUNDS, 1007).split(manifest)
    for split in ("train", "val", "test"):
        assert first.identities(split) == second.identities(split)
    assert first.manifest_hash == second.manifest_hash


def test_seeded_mode_different_seeds_give_different_partitions() -> None:
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    test_sets = {
        frozenset(_seeded_splitter(MAURITIUS_BOUNDS, seed).split(manifest).identities("test"))
        for seed in FOLD_SEEDS
    }
    # 20 folds should give many distinct held-out sets (C(16,2..6) is large).
    assert len(test_sets) >= 15


def test_seeded_mode_shares_the_shuffle_stream_with_exact_mode() -> None:
    """The size draw must not consume the identity-shuffle RNG.

    With bounds pinned so the seeded draw equals the exact-mode sizes, both
    modes must select exactly the same identities — proving the size stream is
    separate and that exact-mode reproducibility is untouched.
    """
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    exact = IdentitySplitter(val_fraction=0.15, test_fraction=0.15, seed=42).split(manifest)
    seeded = IdentitySplitter(
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        size_mode="seeded",
        min_val_identities=2,
        max_val_identities=2,
        min_test_identities=2,
        max_test_identities=2,
    ).split(manifest)
    for split in ("train", "val", "test"):
        assert exact.identities(split) == seeded.identities(split)


def test_seeded_mode_without_caps_pins_sizes_to_the_floor() -> None:
    """A caller who asks for seeded sizing but gives no caps gets the floor.

    ceil(16 * 0.15) = 3, so val and test are 3 each, deterministically.
    """
    manifest = _make_manifest(MAURITIUS_IDENTITIES)
    for seed in FOLD_SEEDS[:5]:
        out = IdentitySplitter(
            val_fraction=0.15, test_fraction=0.15, seed=seed, size_mode="seeded"
        ).split(manifest)
        assert _identity_counts(out) == (10, 3, 3)


def test_seeded_floor_uses_ceil_not_round() -> None:
    """A fraction is a *minimum* in seeded mode, so 12 * 0.15 = 1.8 floors to 2.

    Exact mode rounds it to 2 as well, but for 0.12 the two disagree: round
    gives 1, ceil gives 2. The floor semantics must win.
    """
    manifest = _make_manifest(ROUSETTUS_IDENTITIES)
    seeded = IdentitySplitter(
        val_fraction=0.12, test_fraction=0.12, seed=42, size_mode="seeded"
    ).split(manifest)
    _, n_val, n_test = _identity_counts(seeded)
    assert n_val == 2  # ceil(12 * 0.12) = ceil(1.44) = 2
    assert n_test == 2


def test_seeded_mode_raises_when_the_design_cannot_fit() -> None:
    manifest = _make_manifest(6)
    splitter = IdentitySplitter(
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        size_mode="seeded",
        min_test_identities=3,
        min_val_identities=3,
        min_train_identities=3,
    )
    with pytest.raises(InvalidManifestError, match="cannot satisfy seeded split"):
        splitter.split(manifest)


def test_seeded_cap_is_clamped_so_training_survives() -> None:
    """A cap larger than the pool allows must be trimmed, not violated."""
    manifest = _make_manifest(10)
    for seed in FOLD_SEEDS:
        out = IdentitySplitter(
            val_fraction=0.15,
            test_fraction=0.15,
            seed=seed,
            size_mode="seeded",
            min_test_identities=2,
            max_test_identities=50,  # absurd cap
            min_val_identities=2,
            max_val_identities=2,
            min_train_identities=4,
        ).split(manifest)
        n_train, n_val, n_test = _identity_counts(out)
        assert n_train >= 4
        assert n_val == 2
        assert n_train + n_val + n_test == 10


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_size_mode() -> None:
    with pytest.raises(ValueError, match="size_mode must be"):
        IdentitySplitter(size_mode="random")  # type: ignore[arg-type]


def test_rejects_inverted_identity_bounds() -> None:
    with pytest.raises(ValueError, match="max_test_identities"):
        IdentitySplitter(min_test_identities=5, max_test_identities=2)
    with pytest.raises(ValueError, match="max_val_identities"):
        IdentitySplitter(min_val_identities=4, max_val_identities=1)


def test_rejects_non_positive_bounds() -> None:
    with pytest.raises(ValueError, match="min_test_identities must be >= 1"):
        IdentitySplitter(min_test_identities=0)
    with pytest.raises(ValueError, match="min_train_identities must be >= 1"):
        IdentitySplitter(min_train_identities=0)


# ---------------------------------------------------------------------------
# Identity counts on SplitCounts
# ---------------------------------------------------------------------------


def test_counts_reports_identity_counts_alongside_record_counts() -> None:
    manifest = _make_manifest(MAURITIUS_IDENTITIES, per_id=5)
    splitter = IdentitySplitter(val_fraction=0.15, test_fraction=0.15, seed=42)
    out = splitter.split(manifest)
    counts = splitter.counts(out)

    assert (counts.train_identities, counts.val_identities, counts.test_identities) == (12, 2, 2)
    assert counts.total_identities == MAURITIUS_IDENTITIES
    # 5 records per identity.
    assert counts.train == 12 * 5
    assert counts.val == 2 * 5
    assert counts.test == 2 * 5
    assert counts.total == MAURITIUS_IDENTITIES * 5
