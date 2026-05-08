"""Three-way identity-disjoint train/val/test splitter.

The legacy splitter (``app/siamese_data/data_splitter.py``) is two-way and
TF-coupled. Here we keep just the identity-disjoint logic and produce a
new :class:`bat_core.Manifest` whose records carry the assigned split.

A single bat (``identity``) lives in **exactly one** of train/val/test.
When called twice with the same ``seed`` and the same input identities,
the produced split is bit-identical (Python's stdlib :mod:`random` is
used, with the explicit ``Random(seed)`` instance — no global state).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from bat_core import ImageRecord, Manifest
from bat_core.exceptions import InvalidManifestError

__all__ = ["IdentitySplitter", "SplitCounts"]


@dataclass(frozen=True)
class SplitCounts:
    """Per-split counts — handy for logging / smoke tests."""

    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


class IdentitySplitter:
    """Identity-disjoint 3-way splitter.

    Parameters
    ----------
    val_fraction:
        Fraction of *identities* that go to val (0 < val_fraction < 1).
    test_fraction:
        Fraction of *identities* that go to test (0 < test_fraction < 1).
        ``val_fraction + test_fraction`` must be < 1, leaving the
        remainder to train.
    seed:
        Controls identity shuffling. ``None`` means non-deterministic
        (uses Python's default RNG snapshot).
    min_per_split:
        Each split is guaranteed to receive at least this many identities,
        provided enough are available (the constructor raises otherwise).
    """

    def __init__(
        self,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int | None = 42,
        min_per_split: int = 1,
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

        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = seed
        self.min_per_split = int(min_per_split)

    # ------------------------------------------------------------------
    # Identity-level partitioning
    # ------------------------------------------------------------------

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

        # Shuffle deterministically.
        rng = random.Random(self.seed)
        shuffled = list(ids_sorted)
        rng.shuffle(shuffled)

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
        """Return per-split *record* counts for ``manifest``."""
        train = sum(1 for r in manifest.records if r.split == "train")
        val = sum(1 for r in manifest.records if r.split == "val")
        test = sum(1 for r in manifest.records if r.split == "test")
        return SplitCounts(train=train, val=val, test=test)
