"""Tests for :mod:`bat_stats.model_comparison`.

Where a published reference value exists it is used directly, so these tests
check the implementation against the literature rather than against itself:

* harmonic-mean p -- the ``harmonicmeanp`` package's own worked example.
* Benjamini-Hochberg -- a textbook vector with a hand-checkable answer.
* Nadeau-Bengio -- the exact algebraic relationship to the naive t-statistic.
* Cliff's delta / Hedges' g -- closed-form cases computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from bat_stats.model_comparison import (
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    combine_pvalues,
    compare_groups,
    corrected_resampled_ttest,
    harmonic_mean_p,
    hedges_g,
    hierarchical_bootstrap,
    interpret_delta,
    permutation_p,
    permutation_p_floor,
    tost,
)

# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def test_cliffs_delta_complete_separation() -> None:
    # Every a is below every b -> delta = -1 exactly.
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_identical_samples_is_zero() -> None:
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_hand_computed() -> None:
    # a=[1,3], b=[2,4]: pairs (1,2)- (1,4)- (3,2)+ (3,4)- => (1-3)/4 = -0.5
    assert cliffs_delta([1, 3], [2, 4]) == pytest.approx(-0.5)


def test_cliffs_delta_empty_is_nan() -> None:
    assert np.isnan(cliffs_delta([], [1, 2]))


def test_interpret_delta_thresholds() -> None:
    assert interpret_delta(0.10) == "negligible"
    assert interpret_delta(0.20) == "small"
    assert interpret_delta(0.40) == "medium"
    assert interpret_delta(0.80) == "large"
    assert interpret_delta(-0.80) == "large"  # magnitude only


def test_hedges_g_hand_computed() -> None:
    a = [10.0, 12.0, 14.0]  # mean 12, var 4
    b = [4.0, 6.0, 8.0]  # mean 6,  var 4
    # pooled var = 4 -> d = (12-6)/2 = 3; correction = 1 - 3/(4*6-9) = 1 - 3/15 = 0.8
    assert hedges_g(a, b) == pytest.approx(3.0 * 0.8)


def test_hedges_g_bias_correction_shrinks_estimate() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 8)
    b = rng.normal(0.0, 1.0, 8)
    n1, n2 = len(a), len(b)
    pooled = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    cohens_d = (a.mean() - b.mean()) / np.sqrt(pooled)
    assert abs(hedges_g(a, b)) < abs(cohens_d)


# ---------------------------------------------------------------------------
# Two-group comparison
# ---------------------------------------------------------------------------


def test_compare_groups_reports_separation() -> None:
    result = compare_groups([5, 6, 7, 8, 9], [1, 2, 3, 4])
    assert result.n_a == 5
    assert result.n_b == 4
    assert result.cliffs_delta == pytest.approx(1.0)
    assert result.delta_magnitude == "large"
    assert result.p_value < 0.05
    assert result.median_a == pytest.approx(7.0)
    assert result.median_b == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Equivalence (TOST)
# ---------------------------------------------------------------------------


def test_tost_declares_equivalence_for_near_identical_samples() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(10.0, 1.0, 60)
    b = rng.normal(10.0, 1.0, 60)
    result = tost(a, b, margin=1.0, margin_in_g_units=True)
    assert result.equivalent
    assert result.p_tost < 0.05


def test_tost_rejects_equivalence_for_shifted_samples() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(12.0, 1.0, 60)
    b = rng.normal(10.0, 1.0, 60)
    result = tost(a, b, margin=0.2, margin_in_g_units=True)
    assert not result.equivalent


def test_tost_matches_the_confidence_interval_criterion() -> None:
    # TOST at alpha is equivalent to asking whether the (1-2*alpha) CI lies
    # entirely inside the equivalence bounds. Assert the two agree.
    rng = np.random.default_rng(11)
    for margin in (0.2, 0.5, 1.0, 2.0):
        a = rng.normal(10.2, 1.0, 40)
        b = rng.normal(10.0, 1.0, 40)
        result = tost(a, b, margin=margin, margin_in_g_units=True, alpha=0.05)
        ci_inside = result.margin_low < result.ci_low and result.ci_high < result.margin_high
        assert result.equivalent == ci_inside


def test_tost_raw_units_margin_is_reported() -> None:
    result = tost([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], margin=5.0, margin_in_g_units=False)
    assert result.margin_low == pytest.approx(-5.0)
    assert result.margin_high == pytest.approx(5.0)
    assert "raw" in result.margin_units


def test_tost_needs_two_observations_per_group() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        tost([1.0], [1.0, 2.0], margin=1.0)


# ---------------------------------------------------------------------------
# Nadeau & Bengio corrected resampled t-test
# ---------------------------------------------------------------------------


def test_corrected_ttest_inflates_variance_by_the_exact_factor() -> None:
    """The corrected t must equal the naive t times a known shrink factor.

    naive var = s^2 / n; corrected var = (1/n + n_test/n_train) * s^2, so
    t_corrected / t_naive = sqrt((1/n) / (1/n + n_test/n_train)).
    """
    diffs = [0.03, 0.01, 0.04, 0.02, 0.05, -0.01, 0.02, 0.03]
    n = len(diffs)
    n_train, n_test = 400, 100

    d = np.asarray(diffs, dtype=float)
    naive_t = d.mean() / np.sqrt(d.var(ddof=1) / n)

    result = corrected_resampled_ttest(diffs, n_train=n_train, n_test=n_test)
    expected_ratio = np.sqrt((1.0 / n) / (1.0 / n + n_test / n_train))

    assert result.t_statistic / naive_t == pytest.approx(expected_ratio)
    assert abs(result.t_statistic) < abs(naive_t)  # correction is conservative
    assert result.df == pytest.approx(n - 1)
    assert result.mean_difference == pytest.approx(d.mean())
    assert "nadeau-bengio" in result.correction


def test_corrected_ttest_p_value_exceeds_naive_p_value() -> None:
    from scipy import stats

    diffs = [0.02, 0.03, 0.01, 0.025, 0.015, 0.02]
    d = np.asarray(diffs, dtype=float)
    naive_t = d.mean() / np.sqrt(d.var(ddof=1) / d.size)
    naive_p = 2.0 * stats.t.sf(abs(naive_t), d.size - 1)

    corrected = corrected_resampled_ttest(diffs, n_train=300, n_test=100)
    assert corrected.p_value > naive_p


def test_corrected_ttest_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        corrected_resampled_ttest([0.1], n_train=10, n_test=5)
    with pytest.raises(ValueError, match="zero variance"):
        corrected_resampled_ttest([0.1, 0.1, 0.1], n_train=10, n_test=5)
    with pytest.raises(ValueError, match="must be positive"):
        corrected_resampled_ttest([0.1, 0.2], n_train=0, n_test=5)


# ---------------------------------------------------------------------------
# Permutation p-values
# ---------------------------------------------------------------------------


def test_permutation_p_never_zero_and_hits_its_floor() -> None:
    assert permutation_p(0, 100) == pytest.approx(1.0 / 101.0)
    assert permutation_p(0, 100) == pytest.approx(permutation_p_floor(100))
    assert permutation_p(100, 100) == pytest.approx(1.0)
    assert permutation_p(0, 9999) == pytest.approx(1.0 / 10000.0)


def test_permutation_p_floor_shrinks_with_more_permutations() -> None:
    assert permutation_p_floor(100) > permutation_p_floor(1000) > permutation_p_floor(10_000)


def test_permutation_p_validates_arguments() -> None:
    with pytest.raises(ValueError):
        permutation_p(5, 0)
    with pytest.raises(ValueError):
        permutation_p(101, 100)


# ---------------------------------------------------------------------------
# Harmonic mean p (published reference)
# ---------------------------------------------------------------------------


def test_harmonic_mean_p_matches_published_worked_example() -> None:
    """Reproduce ``harmonicmeanp``'s chromosome-12 example.

    The package vignette reports, for the 312457 p-values of chromosome 12
    within a genome-wide family of L = 6524432 tests: HMP = 8.734522e-4 and
    p.hmp = 1.343897e-3.

    Constructing k p-values all equal to the reported HMP with weights summing
    to 312457 / L reproduces exactly that weighted statistic, so the expected
    output is the published p.hmp.
    """
    n_total = 6_524_432
    n_subset = 312_457
    reported_hmp = 8.734522e-4
    reported_p = 1.343897e-3

    k = 100
    w_sum = n_subset / n_total
    p_values = [reported_hmp] * k
    weights = [w_sum / k] * k

    result = harmonic_mean_p(p_values, weights=weights, n_total=n_total)
    assert result == pytest.approx(reported_p, rel=1e-4)


def test_harmonic_mean_p_is_driven_by_the_smallest_p() -> None:
    # One strong signal among nulls still registers -- the property that makes
    # HMP attractive for combining folds.
    with_signal = harmonic_mean_p([1e-6, 0.9, 0.8, 0.7, 0.6])
    all_null = harmonic_mean_p([0.5, 0.9, 0.8, 0.7, 0.6])
    assert with_signal < all_null


def test_harmonic_mean_p_uniform_nulls_are_not_significant() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        p = rng.uniform(0.0001, 1.0, 20)
        assert harmonic_mean_p(p) > 0.01


def test_harmonic_mean_p_validates_input() -> None:
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        harmonic_mean_p([0.0, 0.5])
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        harmonic_mean_p([1.5])
    with pytest.raises(ValueError, match="no p-values"):
        harmonic_mean_p([])
    with pytest.raises(ValueError, match="n_total must be"):
        harmonic_mean_p([0.1, 0.2, 0.3], n_total=2)
    with pytest.raises(ValueError, match="sum to at most 1"):
        harmonic_mean_p([0.1, 0.2], weights=[0.8, 0.8])


def test_combine_pvalues_hmp_is_more_conservative_than_fisher() -> None:
    # Fisher assumes independence; on correlated folds it is anti-conservative.
    p = [0.02, 0.03, 0.04, 0.02, 0.05]
    assert combine_pvalues(p, method="hmp") > combine_pvalues(p, method="fisher")


def test_combine_pvalues_single_value_passes_through() -> None:
    assert combine_pvalues([0.037]) == pytest.approx(0.037)


def test_combine_pvalues_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        combine_pvalues([0.1, 0.2], method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_textbook_vector() -> None:
    # q_i = min_{j>=i} (n/j) * p_j; for this vector every term equals 0.05.
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert q == pytest.approx([0.05] * 5)


def test_benjamini_hochberg_preserves_input_order() -> None:
    q = benjamini_hochberg([0.05, 0.01, 0.03])
    q_sorted_input = benjamini_hochberg([0.01, 0.03, 0.05])
    assert q[1] == pytest.approx(q_sorted_input[0])
    assert q[2] == pytest.approx(q_sorted_input[1])
    assert q[0] == pytest.approx(q_sorted_input[2])


def test_benjamini_hochberg_never_shrinks_a_p_value() -> None:
    p = [0.001, 0.01, 0.2, 0.5, 0.9]
    q = benjamini_hochberg(p)
    assert np.all(q >= np.asarray(p) - 1e-12)


def test_benjamini_hochberg_empty_input() -> None:
    assert benjamini_hochberg([]).size == 0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_estimate_and_is_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    first = bootstrap_ci(values, n_boot=500, seed=1)
    second = bootstrap_ci(values, n_boot=500, seed=1)
    assert first.ci_low <= first.estimate <= first.ci_high
    assert first.ci_low == pytest.approx(second.ci_low)
    assert first.ci_high == pytest.approx(second.ci_high)


def test_bootstrap_ci_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


def test_hierarchical_bootstrap_detects_a_real_group_difference() -> None:
    rng = np.random.default_rng(5)
    group_a = {f"a{i}": rng.normal(10.0, 0.5, 20).tolist() for i in range(12)}
    group_b = {f"b{i}": rng.normal(7.0, 0.5, 20).tolist() for i in range(12)}
    result = hierarchical_bootstrap(group_a, group_b, n_boot=500, seed=2)
    assert result.estimate == pytest.approx(3.0, abs=0.5)
    assert result.ci_low > 0.0
    assert result.p_value < 0.05


def test_hierarchical_bootstrap_widens_ci_versus_ignoring_clusters() -> None:
    """The point of the two-level bootstrap.

    With strong between-identity variance and near-duplicate observations
    inside each identity, pooling images as if independent fabricates
    precision. The clustered CI must be wider than the naive one.
    """
    rng = np.random.default_rng(13)
    group_a: dict[str, list[float]] = {}
    group_b: dict[str, list[float]] = {}
    for i in range(10):
        # Identity-level offset dominates; within-identity spread is tiny.
        group_a[f"a{i}"] = (rng.normal(10.0, 3.0) + rng.normal(0.0, 0.05, 30)).tolist()
        group_b[f"b{i}"] = (rng.normal(10.0, 3.0) + rng.normal(0.0, 0.05, 30)).tolist()

    clustered = hierarchical_bootstrap(group_a, group_b, n_boot=800, seed=4)

    flat_a = [v for vals in group_a.values() for v in vals]
    flat_b = [v for vals in group_b.values() for v in vals]
    naive = hierarchical_bootstrap(
        {f"one{i}": [v] for i, v in enumerate(flat_a)},
        {f"one{i}": [v] for i, v in enumerate(flat_b)},
        n_boot=800,
        seed=4,
    )

    clustered_width = clustered.ci_high - clustered.ci_low
    naive_width = naive.ci_high - naive.ci_low
    assert clustered_width > naive_width


def test_hierarchical_bootstrap_is_deterministic_for_a_seed() -> None:
    group_a = {"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]}
    group_b = {"p": [0.0, 1.0], "q": [1.0, 2.0]}
    first = hierarchical_bootstrap(group_a, group_b, n_boot=200, seed=9)
    second = hierarchical_bootstrap(group_a, group_b, n_boot=200, seed=9)
    assert first.to_dict() == second.to_dict()


def test_hierarchical_bootstrap_ignores_empty_identities() -> None:
    group_a = {"x": [1.0, 2.0], "empty": []}
    group_b = {"p": [0.0, 1.0]}
    result = hierarchical_bootstrap(group_a, group_b, n_boot=100, seed=1)
    assert np.isfinite(result.estimate)


def test_hierarchical_bootstrap_requires_non_empty_groups() -> None:
    with pytest.raises(ValueError, match="at least one non-empty"):
        hierarchical_bootstrap({"a": []}, {"b": [1.0]}, n_boot=10)
