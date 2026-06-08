"""Manifest builder: walk a fake data tree, collect records."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes


def _stub_quality(_: Path) -> float:
    return 42.0


def test_build_manifest_walks_tree(tmp_path) -> None:
    from bat_data.manifest import build_manifest

    root = tmp_path / "rousettus"
    # Three identities (W, X, Y), spread across split bg/source/aug dirs.
    _touch(root / "video" / "not_augmented" / "random_bg" / "r--W--001.png")
    _touch(root / "video" / "not_augmented" / "random_bg" / "r--W--002.png")
    _touch(root / "video" / "augmented" / "random_bg" / "r--W--002--aug003.png")
    _touch(root / "video" / "not_augmented" / "green_bg" / "r--X--010.png")
    _touch(root / "still" / "not_augmented" / "original_bg" / "r--Y--020.png")
    _touch(root / "video" / "not_augmented" / "random_bg" / "garbage_file.png")  # skipped
    # Hidden file should be skipped.
    _touch(root / "video" / "not_augmented" / "random_bg" / ".DS_Store")

    manifest = build_manifest(root, species="rousettus", compute_quality_fn=_stub_quality)

    assert manifest.manifest_hash
    paths = sorted(str(r.path) for r in manifest.records)
    assert len(paths) == 5
    # Every quality is the stub value.
    assert all(r.quality == 42.0 for r in manifest.records)

    # Background inference works.
    by_path = {r.path.name: r for r in manifest.records}
    assert by_path["r--W--001.png"].background == "random"
    assert by_path["r--X--010.png"].background == "green"
    assert by_path["r--Y--020.png"].background == "original"

    # Source inference works.
    assert by_path["r--Y--020.png"].source == "still"
    assert by_path["r--W--001.png"].source == "video"

    # Augmented flag (filename or directory).
    assert by_path["r--W--002--aug003.png"].augmented is True
    assert by_path["r--W--001.png"].augmented is False

    # Identities propagate.
    assert by_path["r--W--001.png"].identity == "W"


def test_build_manifest_infers_background_from_root_dir(tmp_path) -> None:
    """When --input-dir points directly at a background folder, the background
    token is the root itself and must still be inferred (regression: it used to
    default to 'original' because only sub-paths were inspected).
    """
    from bat_data.manifest import build_manifest

    root = tmp_path / "mauritius" / "video" / "not_augmented" / "green_bg"
    _touch(root / "m--A--001.png")
    _touch(root / "m--B--002.png")

    manifest = build_manifest(root, species="mauritius", compute_quality_fn=_stub_quality)

    assert {r.background for r in manifest.records} == {"green"}
    assert {r.source for r in manifest.records} == {"video"}


def test_build_manifest_missing_root(tmp_path) -> None:
    from bat_data.manifest import build_manifest

    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "does_not_exist", species="rousettus")


def test_build_manifest_strict_mode_raises_on_garbage(tmp_path) -> None:
    from bat_core.exceptions import InvalidManifestError
    from bat_data.manifest import build_manifest

    root = tmp_path / "data"
    _touch(root / "totally_wrong_name.png")
    with pytest.raises(InvalidManifestError):
        build_manifest(
            root,
            species="rousettus",
            compute_quality_fn=_stub_quality,
            skip_unparseable=False,
        )
