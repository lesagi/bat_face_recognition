"""Three-way identity-disjoint train/val/test splitter.

The legacy splitter (``app/siamese_data/data_splitter.py``) is two-way and
TF-coupled. Here we keep just the identity-disjoint logic and produce a
new :class:`bat_core.Manifest` whose records carry the assigned split.

A single bat (``identity``) lives in **exactly one** of train/val/test.
When called twice with the same ``seed`` and the same input identities,
the produced split is bit-identical (Python's stdlib :mod:`random` is
used, with the explicit ``Random(seed)`` instance — no global state).

Two sizing modes
----------------
``size_mode="exact"`` (the default, and unchanged behaviour) reads
``val_fraction`` / ``test_fraction`` as exact targets: the identity counts
are ``round(n * fraction)`` and only *which* identities land where varies
with the seed.

``size_mode="seeded"`` treats the fractions as **floors** and lets the seed
decide the realised sizes as well as the membership. This is what the k-fold
study needs: varying the number of held-out identities across folds turns
"how many test identities do you need for a stable result?" into a
measurable question instead of a fixed assumption.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from bat_core import ImageRecord, Manifest
from bat_core.exceptions import InvalidManifestError

__all__ = ["IdentitySplitter", "SplitCounts", "SplitSizeMode"]

SplitSizeMode = Literal["exact", "seeded"]


@dataclass(frozen=True)
class SplitCounts:
    """Per-split counts — handy for logging / smoke tests.

    ``train``/``val``/``test`` are **record** counts; the ``*_identities``
    fields are identity counts, which is what the split-size sensitivity
    analysis regresses against (and what MLflow logs per run).
    """

    train: int
    val: int
    test: int
    train_identities: int = 0
    val_identities: int = 0
    test_identities: int = 0

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    @property
    def total_identities(self) -> int:
        return self.train_identities + self.val_identities + self.test_identities


class IdentitySplitter:
    """Identity-disjoint 3-way splitter.

    Parameters
    ----------
    val_fraction:
        Fraction of *identities* that go to val (0 < val_fraction < 1). An
        exact target when ``size_mode="exact"``, a **floor** when
        ``size_mode="seeded"``.
    test_fraction:
        Same for test. ``val_fraction + test_fraction`` must be < 1, leaving
        the remainder to train.
    seed:
        Controls identity shuffling, and (in ``seeded`` mode) the split sizes.
        ``None`` means non-deterministic (uses Python's default RNG snapshot).
    min_per_split:
        Each split is guaranteed to receive at least this many identities,
        provided enough are available (the constructor raises otherwise).
    size_mode:
        ``"exact"`` (default) reproduces the historical behaviour exactly:
        ``round(n * fraction)`` identities per split. ``"seeded"`` draws the
        val/test identity counts from the seed, between the fraction-derived
        (or explicit) floors and caps.
    min_val_identities, max_val_identities:
        Explicit integer bounds for val in ``seeded`` mode. When given they
        win over the fraction, which avoids having to reverse-engineer a
        fraction that rounds to the identity count you actually want.
    min_test_identities, max_test_identities:
        Same for test.
    min_train_identities:
        Floor on the training split. Guards the degenerate corner where a
        large test draw starves training.

    Notes
    -----
    In ``seeded`` mode the split sizes are drawn from a *separate* RNG stream
    (``Random(f"{seed}|sizes")``) so the identity-shuffle stream
    (``Random(seed)``) is untouched — ``exact`` mode stays bit-for-bit
    identical to before this mode existed, and the 90 already-published runs
    remain reproducible.
    """

    def __init__(
        self,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int | None = 42,
        min_per_split: int = 1,
        *,
        size_mode: SplitSizeMode = "exact",
        min_val_identities: int | None = None,
        max_val_identities: int | None = None,
        min_test_identities: int | None = None,
        max_test_identities: int | None = None,
        min_train_identities: int = 1,
    ) -> None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction!r}")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction!r}")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction must be < 1; got "
                f"{val_fraction} + {test_fraction} = "
                f"{val_fraction + test_fraction}"
            )
        if min_per_split < 1:
            raise ValueError("min_per_split must be >= 1")
        if size_mode not in ("exact", "seeded"):
            raise ValueError(f"size_mode must be 'exact' or 'seeded'; got {size_mode!r}")
        if min_train_identities < 1:
            raise ValueError("min_train_identities must be >= 1")
        for name, lo, hi in (
            ("val", min_val_identities, max_val_identities),
            ("test", min_test_identities, max_test_identities),
        ):
            if lo is not None and lo < 1:
                raise ValueError(f"min_{name}_identities must be >= 1; got {lo}")
            if hi is not None and hi < 1:
                raise ValueError(f"max_{name}_identities must be >= 1; got {hi}")
            if lo is not None and hi is not None and hi < lo:
                raise ValueError(
                    f"max_{name}_identities ({hi}) must be >= min_{name}_identities ({lo})"
                )

        self.size_mode: SplitSizeMode = size_mode
        self.min_val_identities = min_val_identities
        self.max_val_identities = max_val_identities
        self.min_test_identities = min_test_identities
        self.max_test_identities = max_test_identities
        self.min_train_identities = int(min_train_identities)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = seed
        self.min_per_split = int(min_per_split)

    # ------------------------------------------------------------------
    # Identity-level partitioning
    # ------------------------------------------------------------------

    def _size_bounds(self, n: int, which: str) -> tuple[int, int]:
        """Resolve the (floor, cap) identity count for ``which`` in ``{"val", "test"}``.

        The floor is ``ceil(n * fraction)`` — ceil, not round, because in
        ``seeded`` mode the fraction is a *minimum* guarantee. An explicit
        integer bound overrides it.
        """
        if which == "val":
            fraction, explicit_lo, explicit_hi = (
                self.val_fraction,
                self.min_val_identities,
                self.max_val_identities,
            )
        else:
            fraction, explicit_lo, explicit_hi = (
                self.test_fraction,
                self.min_test_identities,
                self.max_test_identities,
            )

        lo = explicit_lo if explicit_lo is not None else math.ceil(n * fraction)
        lo = max(self.min_per_split, lo, 1)
        # Without an explicit cap the size is pinned to the floor, so a caller
        # who asks for "seeded" without bounds still gets a valid (if constant)
        # split rather than a surprise.
        hi = explicit_hi if explicit_hi is not None else lo
        hi = max(lo, hi)
        return lo, hi

    def _draw_sizes(self, n: int) -> tuple[int, int]:
        """Draw (n_val, n_test) from the seed, respecting all floors and caps.

        Draw order is fixed and part of the reproducibility contract: test
        first, then val, from ``Random(f"{seed}|sizes")``. Caps are clamped so
        the training split keeps at least ``min_train_identities``.
        """
        val_lo, val_hi = self._size_bounds(n, "val")
        test_lo, test_hi = self._size_bounds(n, "test")

        if val_lo + test_lo + self.min_train_identities > n:
            raise InvalidManifestError(
                f"cannot satisfy seeded split of {n} identities: "
                f"val>={val_lo} + test>={test_lo} + train>={self.min_train_identities} "
                f"exceeds {n}"
            )

        size_rng = random.Random(f"{self.seed}|sizes")

        # Test first: it is the quantity the sensitivity study varies, so it
        # gets the unconstrained draw and val absorbs the remainder.
        test_hi = min(test_hi, n - val_lo - self.min_train_identities)
        n_test = size_rng.randint(test_lo, max(test_lo, test_hi))

        val_hi = min(val_hi, n - n_test - self.min_train_identities)
        n_val = size_rng.randint(val_lo, max(val_lo, val_hi))
        return n_val, n_test

    def _partition_identities(
        self, identities: Iterable[str]
    ) -> tuple[set[str], set[str], set[str]]:
        ids_sorted = sorted(set(identities))  # determinism w/ same input
        n = len(ids_sorted)
        if n < 3 * self.min_per_split:
            raise InvalidManifestError(
                f"need at least {3 * self.min_per_split} identities to "
                f"do a 3-way split with min_per_split={self.min_per_split};"
                f" got {n}"
            )

        # Shuffle deterministically. This stream is shared by both size modes
        # so that a given seed always yields the same identity ordering.
        rng = random.Random(self.seed)
        shuffled = list(ids_sorted)
        rng.shuffle(shuffled)

        if self.size_mode == "seeded":
            n_val, n_test = self._draw_sizes(n)
        else:
            n_val = max(self.min_per_split, round(n * self.val_fraction))
            n_test = max(self.min_per_split, round(n * self.test_fraction))
        n_train = n - n_val - n_test

        # Trim if rounding pushed us under min_per_split for train.
        if n_train < self.min_per_split:
            shortfall = self.min_per_split - n_train
            # Steal from whichever has more headroom (val first, then test).
            for _ in range(shortfall):
                if n_val > self.min_per_split:
                    n_val -= 1
                elif n_test > self.min_per_split:
                    n_test -= 1
                else:  # pragma: no cover - guarded by len-check above
                    raise InvalidManifestError("cannot satisfy min_per_split constraints")
            n_train = n - n_val - n_test

        train = set(shuffled[:n_train])
        val = set(shuffled[n_train : n_train + n_val])
        test = set(shuffled[n_train + n_val :])

        # Defensive: assert disjointness.
        if train & val or train & test or val & test:
            raise InvalidManifestError("identity-disjoint invariant violated during partition")
        if not (train and val and test):
            raise InvalidManifestError(
                "one of train/val/test ended up empty; " "check your fractions and identity count"
            )
        return train, val, test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, manifest: Manifest) -> Manifest:
        """Return a *new* manifest whose records carry the assigned split.

        Identities are partitioned once, then every record inherits its
        identity's bucket. The resulting manifest is rehashed (so the
        ``manifest_hash`` differs from the input).
        """
        identities = {r.identity for r in manifest.records}
        train_ids, val_ids, test_ids = self._partition_identities(identities)

        new_records: list[ImageRecord] = []
        for r in manifest.records:
            if r.identity in train_ids:
                split: str = "train"
            elif r.identity in val_ids:
                split = "val"
            elif r.identity in test_ids:
                split = "test"
            else:  # pragma: no cover - all identities accounted for
                raise InvalidManifestError(f"identity {r.identity!r} not assigned to any split")
            new_records.append(r.model_copy(update={"split": split}))  # type: ignore[arg-type]

        new_manifest = Manifest.from_records(new_records)
        # Hard guard before returning.
        new_manifest.assert_identity_disjoint()
        return new_manifest

    def counts(self, manifest: Manifest) -> SplitCounts:
        """Return per-split record **and** identity counts for ``manifest``."""
        train = sum(1 for r in manifest.records if r.split == "train")
        val = sum(1 for r in manifest.records if r.split == "val")
        test = sum(1 for r in manifest.records if r.split == "test")
        return SplitCounts(
            train=train,
            val=val,
            test=test,
            train_identities=len(manifest.identities("train")),
            val_identities=len(manifest.identities("val")),
            test_identities=len(manifest.identities("test")),
        )
