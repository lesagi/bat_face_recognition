"""Regression tests for saliency checkpoint resolution.

These exist because the original resolver silently returned the WRONG
checkpoint. It filtered on species, model family and input edge but not on
background, then broke ties with ``max(..., key=mtime)``. Once Phase 3 wrote
``green-dilated`` runs, asking for an ``original`` model returned a dilated
checkpoint — and because the dilated and stock backbones have identical
parameter shapes, ``load_state_dict`` accepted it without complaint. Every
stride and receptive field would have been wrong with no error anywhere.

Nothing downstream can catch that, so it has to be caught here.
"""

from __future__ import annotations

import pytest
from species_saliency_maps import find_run_dir_for_edge


def _make_run(root, leaf: str, ckpt: str = "best_model_roc_auc.pt"):
    d = root / "outputs" / "runs" / "2026-01-01" / leaf
    d.mkdir(parents=True, exist_ok=True)
    (d / ckpt).write_bytes(b"")
    return d


def test_background_token_is_not_a_substring_match(tmp_path, monkeypatch):
    """`green` must not match `green-dilated` — the collision that caused the bug."""
    monkeypatch.chdir(tmp_path)
    _make_run(tmp_path, "mauritius_video_green-dilated_arcface_arcface_e320_s42")
    assert find_run_dir_for_edge("mauritius", "arcface", 320, "green") is None


def test_resolves_the_matching_background(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_run(tmp_path, "mauritius_video_green-dilated_arcface_arcface_e320_s42")
    wanted = _make_run(tmp_path, "mauritius_video_green_arcface_arcface_e320_s0")
    got = find_run_dir_for_edge("mauritius", "arcface", 320, "green")
    assert got is not None and got.resolve() == wanted.resolve()


def test_ambiguity_raises_instead_of_guessing(tmp_path, monkeypatch):
    """Ten folds all match legitimately; picking by mtime is not an answer."""
    monkeypatch.chdir(tmp_path)
    for fold in range(3):
        _make_run(tmp_path, f"mauritius_video_green_arcface_arcface_e320_s{fold}")
    with pytest.raises(SystemExit, match="ambiguous run directory"):
        find_run_dir_for_edge("mauritius", "arcface", 320, "green")


def test_edge_token_still_discriminates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_run(tmp_path, "mauritius_video_green_arcface_arcface_e112_s0")
    assert find_run_dir_for_edge("mauritius", "arcface", 320, "green") is None


def test_directory_without_a_checkpoint_is_not_a_candidate(tmp_path, monkeypatch):
    """Why the live tree is unambiguous: the pipeline deletes checkpoints."""
    monkeypatch.chdir(tmp_path)
    kept = _make_run(tmp_path, "mauritius_video_green_arcface_arcface_e320_s0")
    bare = tmp_path / "outputs" / "runs" / "2026-01-01" / "mauritius_video_green_arcface_arcface_e320_s1"
    bare.mkdir(parents=True)
    got = find_run_dir_for_edge("mauritius", "arcface", 320, "green")
    assert got is not None and got.resolve() == kept.resolve()
