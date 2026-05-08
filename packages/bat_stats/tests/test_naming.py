"""Tests for the experiment-naming helper.

The naming bug fix is the central feature of bat_stats: every output filename
must encode ``species_source_background_model_loss``.  These unit tests cover
the helper in isolation; the regression test in
``test_visualizer_naming.py`` covers the end-to-end "no plot escapes with a
generic name" guarantee.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from bat_stats.naming import (
    EXPERIMENT_NAME_FIELDS,
    build_axis_label,
    build_filename,
    build_title_suffix,
    experiment_name,
    extract_components,
)


def test_experiment_name_dict_cfg(cfg, expected_components) -> None:
    name = experiment_name(cfg)
    assert name == "rousettus_video_random_arcface_arcface"
    for value in expected_components.values():
        assert value in name


def test_experiment_name_namespace_cfg(cfg_namespace) -> None:
    assert experiment_name(cfg_namespace) == "mauritius_still_green_siamese_bce"


def test_extract_components_returns_all_fields(cfg, expected_components) -> None:
    out = extract_components(cfg)
    assert set(out.keys()) == set(EXPERIMENT_NAME_FIELDS)
    assert out == expected_components


def test_extract_components_handles_missing_fields() -> None:
    out = extract_components({})
    for field in EXPERIMENT_NAME_FIELDS:
        assert out[field] == "unknown"


def test_extract_components_slugifies_unsafe_characters() -> None:
    cfg = {
        "data": {"species": "Rousettus / Bat", "source": "VIDEO", "background": "random_bg"},
        "model": {"name": "arc face"},
        "loss": {"name": "arcface@v2"},
    }
    out = extract_components(cfg)
    assert out["species"] == "rousettus-bat"
    assert out["source"] == "video"
    assert out["background"] == "random-bg"
    assert out["model"] == "arc-face"
    assert out["loss"] == "arcfacev2"


def test_build_filename_appends_extension_and_signature(cfg) -> None:
    fname = build_filename("null_dist_f1", cfg)
    assert fname == "null_dist_f1__rousettus_video_random_arcface_arcface.png"


def test_build_filename_pdf_suffix(cfg) -> None:
    fname = build_filename("summary", cfg, suffix=".pdf")
    assert fname.endswith(".pdf")
    assert "rousettus_video_random_arcface_arcface" in fname


def test_build_filename_normalises_suffix(cfg) -> None:
    fname = build_filename("summary", cfg, suffix="csv")
    assert fname.endswith(".csv")


def test_build_filename_rejects_empty_stem(cfg) -> None:
    with pytest.raises(ValueError):
        build_filename("", cfg)


def test_build_title_suffix_includes_all_components(cfg, expected_components) -> None:
    suffix = build_title_suffix(cfg)
    for value in expected_components.values():
        assert value in suffix


def test_build_axis_label_includes_experiment_name(cfg) -> None:
    label = build_axis_label("F1", cfg)
    assert label.startswith("F1")
    assert "rousettus_video_random_arcface_arcface" in label


def test_top_level_keys_take_precedence_over_unknown_nested() -> None:
    """If both top-level and nested versions exist, top-level wins."""

    cfg = {"species": "mauritius", "source": "video", "background": "green"}
    name = experiment_name(cfg)
    assert "mauritius" in name
    assert "video" in name
    assert "green" in name


def test_dataclass_like_object_with_name_attr() -> None:
    """``model.name`` style access works for SimpleNamespace, dataclasses, etc."""

    cfg = SimpleNamespace(
        species="rousettus",
        source="video",
        background="original",
        model=SimpleNamespace(name="adaface"),
        loss=SimpleNamespace(name="adaface"),
    )
    assert experiment_name(cfg) == "rousettus_video_original_adaface_adaface"
