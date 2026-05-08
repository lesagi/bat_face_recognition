"""Manifest construction utilities.

Walks a data directory, parses filenames according to the
``(type)--(class)--(id)[--aug###].png`` convention used across the
project, computes a per-image quality score (Laplacian variance), and
emits a :class:`bat_core.Manifest`.

The directory layout assumed (matches the legacy ``app/`` layout):

    <root>/<species>/<source>/<aug_segment>/<background>/<files>

where:
    * ``<source>``      ∈ {``video``, ``still``}
    * ``<aug_segment>`` ∈ {``not_augmented``, ``augmented``}
        (the on-disk flag; per-file ``--aug###`` suffix takes precedence)
    * ``<background>``  ∈ {``green``, ``random``, ``original``}

The walker accepts deviations gracefully: if the path does not match the
expected layout, the file is still indexed by parsing as much as
possible from the filename and falling back to defaults documented on
:func:`build_manifest`.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from bat_core import ImageRecord, Manifest
from bat_core.exceptions import InvalidManifestError

__all__ = [
    "FILENAME_PATTERN",
    "ParsedFilename",
    "parse_filename",
    "group_files_by_class",
    "is_augmented_file",
    "compute_quality",
    "build_manifest",
    "manifest_to_csv",
    "manifest_from_csv",
]


# ---------------------------------------------------------------------------
# Filename parsing (ported from app/utils/filename_parser.py)
# ---------------------------------------------------------------------------

# Pattern: (type)--(class)--(id)[--aug###]
# - type: short token (e.g. "r" for rousettus)
# - class: identity (alphanumeric)
# - id: capture id (allows ``\d+.\d+`` decimal style)
# - optional --aug### suffix where ### is exactly three digits
FILENAME_PATTERN = re.compile(
    r"^(?P<type>\w+)--(?P<class>\w+)--(?P<id>\w+(?:\.\d+)?)"
    r"(?P<aug_suffix>--aug(?P<aug_id>\d{3}))?$"
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ParsedFilename:
    """Lightweight result type for :func:`parse_filename`."""

    __slots__ = ("type_", "class_name", "id_", "aug_id")

    def __init__(
        self,
        type_: str,
        class_name: str,
        id_: str,
        aug_id: str | None,
    ) -> None:
        self.type_ = type_
        self.class_name = class_name
        self.id_ = id_
        self.aug_id = aug_id

    @property
    def is_augmented(self) -> bool:
        return self.aug_id is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ParsedFilename(type={self.type_!r}, class={self.class_name!r}, "
            f"id={self.id_!r}, aug_id={self.aug_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedFilename):
            return NotImplemented
        return (
            self.type_ == other.type_
            and self.class_name == other.class_name
            and self.id_ == other.id_
            and self.aug_id == other.aug_id
        )


def parse_filename(filename: str) -> ParsedFilename | None:
    """Parse a filename into ``(type, class, id, aug_id)`` components.

    Returns ``None`` when the filename does not match the expected
    pattern.
    """
    name_without_ext = os.path.splitext(filename)[0]
    match = FILENAME_PATTERN.match(name_without_ext)
    if match is None:
        return None
    g = match.groupdict()
    return ParsedFilename(
        type_=g["type"],
        class_name=g["class"],
        id_=g["id"],
        aug_id=g.get("aug_id"),
    )


def is_augmented_file(filename: str) -> bool:
    """Return ``True`` if ``filename`` carries the ``--aug###`` suffix."""
    parsed = parse_filename(filename)
    return parsed is not None and parsed.is_augmented


def group_files_by_class(
    file_paths: Iterable[str | os.PathLike[str]],
) -> dict[str, dict[str, list[str]]]:
    """Group files by parsed class then capture id.

    Mirrors ``app/utils/filename_parser.py:group_files_by_class`` but
    drops the warning print so it is safe to call from library code.
    """
    class_files: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for file_path in file_paths:
        path_str = os.fspath(file_path)
        parsed = parse_filename(os.path.basename(path_str))
        if parsed is None:
            continue
        class_files[parsed.class_name][parsed.id_].append(path_str)
    # Convert nested defaultdicts to plain dicts for stable serialization.
    return {k: dict(v) for k, v in class_files.items()}


# ---------------------------------------------------------------------------
# Quality (Laplacian variance) - cv2 only imported when needed
# ---------------------------------------------------------------------------


def compute_quality(image_path: str | os.PathLike[str]) -> float:
    """Compute the Laplacian-variance focus measure for ``image_path``.

    Uses OpenCV. Returns ``0.0`` if the image cannot be loaded so the
    Manifest builder can continue past corrupt files (build_manifest
    logs a warning in that branch).
    """
    import cv2  # local import: opencv-python is heavy

    img = cv2.imread(os.fspath(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


# ---------------------------------------------------------------------------
# Directory layout inference
# ---------------------------------------------------------------------------


_BACKGROUND_DIR_MAP = {
    "green_bg": "green",
    "green": "green",
    "random_bg": "random",
    "random": "random",
    "original_bg": "original",
    "original": "original",
    "no_bg": "original",
}


def _infer_background(parts: tuple[str, ...]) -> str:
    for p in parts:
        token = p.lower()
        if token in _BACKGROUND_DIR_MAP:
            return _BACKGROUND_DIR_MAP[token]
    return "original"


def _infer_source(parts: tuple[str, ...]) -> str:
    lowered = {p.lower() for p in parts}
    if "video" in lowered:
        return "video"
    if "still" in lowered or "stills" in lowered:
        return "still"
    return "video"


def _infer_dir_augmented(parts: tuple[str, ...]) -> bool | None:
    """Return ``True``/``False`` if directory says augmented; ``None`` if unknown."""
    lowered = {p.lower() for p in parts}
    if "augmented" in lowered:
        return True
    if "not_augmented" in lowered:
        return False
    return None


def _iter_image_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith(".") or fn.startswith("._"):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                yield Path(dirpath) / fn


# ---------------------------------------------------------------------------
# Manifest build / CSV round-trip
# ---------------------------------------------------------------------------


def build_manifest(
    root_dir: str | os.PathLike[str],
    species: str,
    *,
    compute_quality_fn=compute_quality,
    skip_unparseable: bool = True,
    default_split: str = "train",
) -> Manifest:
    """Walk ``root_dir`` and build a :class:`bat_core.Manifest`.

    ``compute_quality_fn`` is exposed so tests (and callers without
    OpenCV available) can substitute a lightweight stub. ``species`` is
    stamped onto every record.

    Notes
    -----
    * ``split`` is set to ``default_split`` for every record. The
      :class:`bat_data.splitter.IdentitySplitter` produces a manifest
      with the proper train/val/test partition.
    * ``background`` and ``source`` are inferred from path components
      (see ``_infer_background`` / ``_infer_source``).
    * ``augmented`` is ``True`` when either the filename has ``--aug###``
      *or* the file lives inside an ``augmented/`` directory.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    records: list[ImageRecord] = []
    for path in sorted(_iter_image_files(root)):
        parsed = parse_filename(path.name)
        if parsed is None:
            if skip_unparseable:
                continue
            raise InvalidManifestError(f"unparseable filename: {path.name}")

        rel_parts = path.relative_to(root).parts[:-1]
        background = _infer_background(rel_parts)
        source = _infer_source(rel_parts)
        dir_aug = _infer_dir_augmented(rel_parts)
        # Filename-level aug always wins; otherwise fall back to dir hint
        # (or False when neither says anything).
        augmented = parsed.is_augmented or bool(dir_aug)

        try:
            quality = float(compute_quality_fn(path))
        except Exception:  # pragma: no cover - defensive
            quality = 0.0
        if quality < 0.0:
            quality = 0.0

        records.append(
            ImageRecord(
                path=path,
                identity=parsed.class_name,
                species=species,  # type: ignore[arg-type]
                background=background,  # type: ignore[arg-type]
                source=source,  # type: ignore[arg-type]
                augmented=augmented,
                split=default_split,  # type: ignore[arg-type]
                quality=quality,
            )
        )

    return Manifest.from_records(records)


_CSV_COLUMNS = [
    "path",
    "identity",
    "species",
    "background",
    "source",
    "augmented",
    "split",
    "quality",
]


def manifest_to_csv(
    manifest: Manifest, csv_path: str | os.PathLike[str]
) -> Path:
    """Persist ``manifest`` to ``csv_path`` (always writes the columns
    in the canonical order). The companion ``<csv>.hash`` sidecar holds
    the computed manifest hash so :func:`manifest_from_csv` can verify
    round-trip integrity.
    """
    rows = [
        {
            "path": str(r.path),
            "identity": r.identity,
            "species": r.species,
            "background": r.background,
            "source": r.source,
            "augmented": bool(r.augmented),
            "split": r.split,
            "quality": float(r.quality),
        }
        for r in manifest.records
    ]
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sidecar = out.with_suffix(out.suffix + ".hash")
    sidecar.write_text(manifest.manifest_hash, encoding="utf-8")
    return out


def manifest_from_csv(csv_path: str | os.PathLike[str]) -> Manifest:
    """Read a CSV produced by :func:`manifest_to_csv` back into a
    :class:`Manifest`. The recomputed ``manifest_hash`` is compared
    against the sidecar (when present) and an
    :class:`InvalidManifestError` is raised on mismatch.
    """
    csv_p = Path(csv_path)
    df = pd.read_csv(csv_p)
    missing = [c for c in _CSV_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidManifestError(
            f"manifest CSV missing columns: {missing}"
        )

    records: list[ImageRecord] = []
    for row in df.itertuples(index=False):
        record_dict = {c: getattr(row, c) for c in _CSV_COLUMNS}
        records.append(
            ImageRecord(
                path=Path(str(record_dict["path"])),
                identity=str(record_dict["identity"]),
                species=str(record_dict["species"]),  # type: ignore[arg-type]
                background=str(record_dict["background"]),  # type: ignore[arg-type]
                source=str(record_dict["source"]),  # type: ignore[arg-type]
                augmented=bool(record_dict["augmented"]),
                split=str(record_dict["split"]),  # type: ignore[arg-type]
                quality=float(record_dict["quality"]),
            )
        )

    manifest = Manifest.from_records(records)

    sidecar = csv_p.with_suffix(csv_p.suffix + ".hash")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip()
        if expected and expected != manifest.manifest_hash:
            raise InvalidManifestError(
                f"manifest CSV hash mismatch: "
                f"sidecar={expected[:16]}... got={manifest.manifest_hash[:16]}..."
            )
    return manifest


# ---------------------------------------------------------------------------
# Manifest convenience helpers (attached to Manifest at import time)
# ---------------------------------------------------------------------------


def _manifest_to_csv_method(
    self: Manifest, path: str | os.PathLike[str]
) -> Path:
    """``Manifest.to_csv(path)`` shortcut."""
    return manifest_to_csv(self, path)


def _manifest_from_csv_classmethod(
    cls: type[Manifest], path: str | os.PathLike[str]
) -> Manifest:
    """``Manifest.from_csv(path)`` shortcut."""
    return manifest_from_csv(path)


# Attach helpers to bat_core.Manifest without modifying bat_core source.
# These names are the documented public surface for CSV round-trip.
if not hasattr(Manifest, "to_csv"):
    Manifest.to_csv = _manifest_to_csv_method  # type: ignore[attr-defined]
if not hasattr(Manifest, "from_csv"):
    Manifest.from_csv = classmethod(_manifest_from_csv_classmethod)  # type: ignore[attr-defined]
