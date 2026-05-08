"""Filename parser round-trip + edge cases."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from bat_data.manifest import group_files_by_class, is_augmented_file, parse_filename  # noqa: E402


@pytest.mark.parametrize(
    "filename, expected_type, expected_class, expected_id, expected_aug",
    [
        ("r--W--00012.png", "r", "W", "00012", None),
        ("r--W--00012--aug003.png", "r", "W", "00012", "003"),
        ("type1--ABC--id_5.png", "type1", "ABC", "id_5", None),
        ("r--W--00012.45.png", "r", "W", "00012.45", None),
        ("r--W--00012.45--aug999.png", "r", "W", "00012.45", "999"),
        ("r--W--abc123.png", "r", "W", "abc123", None),
    ],
)
def test_parse_known_patterns(
    filename, expected_type, expected_class, expected_id, expected_aug
) -> None:
    result = parse_filename(filename)
    assert result is not None
    assert result.type_ == expected_type
    assert result.class_name == expected_class
    assert result.id_ == expected_id
    assert result.aug_id == expected_aug
    assert result.is_augmented == (expected_aug is not None)


@pytest.mark.parametrize(
    "filename",
    [
        "no_separator.png",
        "only-one--separator.png",  # only one ``--``
        "r--W--00012--aug12.png",  # aug must be 3 digits
        "r--W--00012--aug12345.png",  # aug must be 3 digits
        "--W--00012.png",  # missing type
        "r----00012.png",  # empty class
        "",
    ],
)
def test_parse_rejects_bad_patterns(filename: str) -> None:
    assert parse_filename(filename) is None


def test_is_augmented_file() -> None:
    assert is_augmented_file("r--W--1--aug001.png") is True
    assert is_augmented_file("r--W--1.png") is False
    assert is_augmented_file("garbage.png") is False


def test_group_files_by_class_handles_nested_paths() -> None:
    files = [
        "/data/r--W--001.png",
        "/data/r--W--002.png",
        "/data/r--W--002--aug003.png",
        "/data/r--X--010.png",
        "/data/garbage.png",  # silently skipped
    ]
    grouped = group_files_by_class(files)
    assert set(grouped.keys()) == {"W", "X"}
    assert set(grouped["W"].keys()) == {"001", "002"}
    # The base + aug share the same id (id == "002")
    assert len(grouped["W"]["002"]) == 2
    assert grouped["X"]["010"] == ["/data/r--X--010.png"]


def test_group_files_returns_plain_dicts() -> None:
    """Defaultdicts leak surprises through serialization; ensure we
    return plain dicts."""
    grouped = group_files_by_class(["/x/r--C--1.png"])
    assert type(grouped) is dict
    assert type(grouped["C"]) is dict
