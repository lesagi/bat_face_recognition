"""Helpers to derive a gallery / probe split from a Manifest partition.

Given the val or test split of a :class:`bat_core.Manifest` and an embedding
function, return ``(gallery_embedding, probe_embedding)``.

Default split policy: the first ``gallery_size`` images per identity (sorted
by path for determinism) become the gallery; the rest are probes. Identities
with too few samples are skipped (logged via a return-side ``skipped`` list).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from bat_core import Embedding, ImageRecord, Manifest, Split


@dataclass(frozen=True)
class GalleryProbeSplit:
    """Records selected for the gallery and probe sets, plus skipped identities."""

    gallery_records: tuple[ImageRecord, ...]
    probe_records: tuple[ImageRecord, ...]
    skipped_identities: tuple[str, ...]


def split_gallery_probe(
    records: Iterable[ImageRecord],
    gallery_size: int = 1,
    min_probe_size: int = 1,
) -> GalleryProbeSplit:
    """Group ``records`` by identity and pick the gallery / probe partition.

    Parameters
    ----------
    records: image records to partition.
    gallery_size: number of images per identity that go to the gallery.
        The first ``gallery_size`` records (sorted by path) are taken.
    min_probe_size: identities whose remaining (probe) sample count drops
        below this are skipped entirely.
    """
    if gallery_size < 1:
        raise ValueError(f"gallery_size must be >= 1, got {gallery_size}")
    if min_probe_size < 1:
        raise ValueError(f"min_probe_size must be >= 1, got {min_probe_size}")

    by_id: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_id[r.identity].append(r)

    gallery: list[ImageRecord] = []
    probe: list[ImageRecord] = []
    skipped: list[str] = []
    for identity, items in by_id.items():
        items_sorted = sorted(items, key=lambda r: str(r.path))
        if len(items_sorted) < gallery_size + min_probe_size:
            skipped.append(identity)
            continue
        gallery.extend(items_sorted[:gallery_size])
        probe.extend(items_sorted[gallery_size:])

    return GalleryProbeSplit(
        gallery_records=tuple(gallery),
        probe_records=tuple(probe),
        skipped_identities=tuple(sorted(skipped)),
    )


def split_manifest_for_identification(
    manifest: Manifest,
    split: Split,
    gallery_size: int = 1,
    min_probe_size: int = 1,
) -> GalleryProbeSplit:
    """Convenience: pull records from a single split and partition them."""
    return split_gallery_probe(
        manifest.filter_split(split),
        gallery_size=gallery_size,
        min_probe_size=min_probe_size,
    )


EmbedFn = Callable[[list[Path]], object]
"""Signature for an embedding function: list of paths → 2-D torch.Tensor."""


def materialize_embedding(
    records: Iterable[ImageRecord],
    embed_fn: EmbedFn,
) -> Embedding:
    """Run ``embed_fn`` on the record paths and wrap the result.

    The embedding function is expected to return a 2-D ``torch.Tensor`` of
    shape ``(N, D)`` aligned with the input order; we attach the matching
    identity list and return a :class:`bat_core.Embedding`.
    """
    rec_list = list(records)
    paths = [r.path for r in rec_list]
    identities = tuple(r.identity for r in rec_list)
    tensor = embed_fn(paths)
    return Embedding(tensor=tensor, identities=identities)


__all__ = [
    "EmbedFn",
    "GalleryProbeSplit",
    "materialize_embedding",
    "split_gallery_probe",
    "split_manifest_for_identification",
]
