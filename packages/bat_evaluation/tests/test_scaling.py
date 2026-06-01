"""Tests for the monotone score-scaling helpers."""

from __future__ import annotations

import numpy as np
import pytest
from bat_evaluation.scaling import arccos_scale, inv_neg_log_1_minus, neg_log_1_minus
from sklearn.metrics import roc_auc_score


def test_neg_log_1_minus_monotone_increasing():
    s = np.array([-0.5, 0.0, 0.5, 0.9, 0.99, 0.999, 0.9999])
    t = neg_log_1_minus(s)
    assert np.all(np.diff(t) > 0), f"transform must be monotone increasing, got {t}"


def test_neg_log_1_minus_handles_s_eq_one_without_inf():
    # ``s = 1`` would naively produce ``-log(0) = inf``; the ``eps`` floor
    # must clamp it to a finite value.
    t = neg_log_1_minus(np.array([1.0, 1.0]))
    assert np.all(np.isfinite(t)), f"expected finite, got {t}"
    assert t[0] > 25.0, "with eps=1e-12, t(1.0) should be ~27.6"


def test_neg_log_1_minus_round_trip():
    # ``s in [-0.9, 1 - eps]`` should round-trip cleanly through the
    # ``neg_log_1_minus`` / ``inv_neg_log_1_minus`` pair.
    s = np.array([-0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999_999])
    s_back = inv_neg_log_1_minus(neg_log_1_minus(s))
    np.testing.assert_allclose(s_back, s, atol=1e-9)


def test_neg_log_1_minus_preserves_roc_auc():
    # ROC-AUC is invariant to any monotone-increasing transformation.
    rng = np.random.default_rng(0)
    pos = rng.normal(0.97, 0.01, size=200).clip(min=-1.0, max=1.0 - 1e-9)
    neg = rng.normal(0.93, 0.01, size=400).clip(min=-1.0, max=1.0 - 1e-9)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    raw_auc = roc_auc_score(y_true, y_score)
    scaled_auc = roc_auc_score(y_true, neg_log_1_minus(y_score))
    assert raw_auc == pytest.approx(scaled_auc, abs=1e-12)


def test_neg_log_1_minus_visible_spread_when_raw_clusters_near_one():
    # Concretely shows the motivation: raw scores within 1e-7 of 1.0 are
    # crowded; the transform spreads them by ~one order of magnitude.
    raw = np.array([1 - 1e-7, 1 - 1e-8, 1 - 1e-9])
    scaled = neg_log_1_minus(raw)
    raw_spread = raw.max() - raw.min()
    scaled_spread = scaled.max() - scaled.min()
    assert raw_spread < 1e-6
    assert scaled_spread > 4.0  # ln(100) ≈ 4.6 spans 2 decades


def test_arccos_scale_monotone_decreasing():
    s = np.array([-1.0, -0.5, 0.0, 0.5, 0.9, 0.999])
    t = arccos_scale(s)
    assert np.all(np.diff(t) < 0), f"arccos must be monotone decreasing, got {t}"


def test_arccos_scale_handles_out_of_range_inputs():
    # Floating-point cosine sims may marginally exceed +/-1; clipping must
    # keep us inside arccos's domain.
    t = arccos_scale(np.array([1.0 + 1e-9, -1.0 - 1e-9, 0.5]))
    assert np.all(np.isfinite(t))
    assert t[0] == pytest.approx(0.0)
    assert t[1] == pytest.approx(np.pi)


def test_arccos_scale_auc_inversion():
    # Decreasing transformation flips AUC: ``auc(y, arccos(s)) == 1 - auc(y, s)``.
    rng = np.random.default_rng(7)
    pos = rng.normal(0.7, 0.1, size=200).clip(min=-1.0, max=1.0 - 1e-9)
    neg = rng.normal(0.3, 0.1, size=200).clip(min=-1.0, max=1.0 - 1e-9)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    raw_auc = roc_auc_score(y_true, y_score)
    arccos_auc = roc_auc_score(y_true, arccos_scale(y_score))
    assert arccos_auc == pytest.approx(1.0 - raw_auc, abs=1e-12)
