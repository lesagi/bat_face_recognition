"""Statistical primitives for comparing models, folds, and species.

Everything here exists because the project's default tests are wrong for its
data. Three recurring problems:

1. **Cross-validation folds are not independent.** They share training data, so
   a plain paired t-test over 20 folds understates the variance and inflates the
   Type I error rate. :func:`corrected_resampled_ttest` applies the Nadeau &
   Bengio variance correction.
2. **Images are not independent.** 520 vs 1092 images clustered inside 16 vs 12
   individuals means a per-image test is pseudoreplication and will report
   p < 1e-30 for almost any comparison. The unit of analysis must be the
   identity — :func:`hierarchical_bootstrap` resamples identities first and
   observations second.
3. **"No significant difference" is not a result.** A non-significant p is not
   evidence of equivalence. :func:`tost` tests equivalence against a declared
   margin, which is what a claim of dataset parity actually requires.

Plus the aggregation machinery: :func:`combine_pvalues` (harmonic-mean p for
dependent tests, Fisher for independent ones), :func:`benjamini_hochberg` for
multiplicity across cells, and :func:`permutation_p` so a Monte-Carlo
permutation p-value is never reported as zero or below its own resolution floor.

Effect sizes (:func:`cliffs_delta`, :func:`hedges_g`) accompany every test — with
12-20 units per group, a p-value alone says almost nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats
from scipy.special import digamma

__all__ = [
    "BootstrapResult",
    "ComparisonResult",
    "PairedResult",
    "EquivalenceResult",
    "TTestResult",
    "benjamini_hochberg",
    "bootstrap_ci",
    "cliffs_delta",
    "combine_pvalues",
    "compare_groups",
    "corrected_resampled_ttest",
    "harmonic_mean_p",
    "hedges_g",
    "hierarchical_bootstrap",
    "interpret_delta",
    "paired_wilcoxon",
    "permutation_p",
    "permutation_p_floor",
    "tost",
]


# ---------------------------------------------------------------------------
# Result containers (dataclass-only output, per the project convention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a two-group comparison."""

    n_a: int
    n_b: int
    median_a: float
    median_b: float
    mann_whitney_u: float
    p_value: float
    cliffs_delta: float
    delta_magnitude: str
    hedges_g: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "median_a": self.median_a,
            "median_b": self.median_b,
            "mann_whitney_u": self.mann_whitney_u,
            "p_value": self.p_value,
            "cliffs_delta": self.cliffs_delta,
            "delta_magnitude": self.delta_magnitude,
            "hedges_g": self.hedges_g,
        }


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired within-unit comparison.

    Separate from :class:`ComparisonResult` because the question is different.
    That one asks whether two *groups* differ; this asks whether a *change*
    applied to the same units is non-zero. Using the unpaired test on paired data
    throws away the pairing and loses most of the power.
    """

    n_pairs: int
    median_before: float
    median_after: float
    median_difference: float
    wilcoxon_w: float
    p_value: float
    p_value_one_sided: float
    matched_pairs_rank_biserial: float
    effect_magnitude: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "n_pairs": self.n_pairs,
            "median_before": self.median_before,
            "median_after": self.median_after,
            "median_difference": self.median_difference,
            "wilcoxon_w": self.wilcoxon_w,
            "p_value": self.p_value,
            "p_value_one_sided": self.p_value_one_sided,
            "matched_pairs_rank_biserial": self.matched_pairs_rank_biserial,
            "effect_magnitude": self.effect_magnitude,
        }


@dataclass(frozen=True)
class EquivalenceResult:
    """Outcome of a two one-sided tests (TOST) equivalence test."""

    mean_difference: float
    margin_low: float
    margin_high: float
    p_lower: float
    p_upper: float
    p_tost: float
    ci_low: float
    ci_high: float
    equivalent: bool
    margin_units: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "mean_difference": self.mean_difference,
            "margin_low": self.margin_low,
            "margin_high": self.margin_high,
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_tost": self.p_tost,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "equivalent": self.equivalent,
            "margin_units": self.margin_units,
        }


@dataclass(frozen=True)
class TTestResult:
    """Outcome of a (possibly variance-corrected) t-test."""

    mean_difference: float
    t_statistic: float
    df: float
    p_value: float
    correction: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mean_difference": self.mean_difference,
            "t_statistic": self.t_statistic,
            "df": self.df,
            "p_value": self.p_value,
            "correction": self.correction,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a bootstrap: point estimate, CI, and a two-sided p-value."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    confidence: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_boot": self.n_boot,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    A non-parametric effect size that needs no distributional assumption —
    appropriate for the small, skewed per-identity samples here.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def interpret_delta(delta: float) -> str:
    """Romano et al. (2006) magnitude labels for Cliff's delta."""
    d = abs(delta)
    if np.isnan(d):
        return "undefined"
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Hedges' g — Cohen's d with the small-sample bias correction.

    The correction matters: with n around 12-16 per group, uncorrected d
    overstates the effect by roughly 5%.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return float("nan")
    d = (x.mean() - y.mean()) / np.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(d * correction)


# ---------------------------------------------------------------------------
# Two-group comparison
# ---------------------------------------------------------------------------


def paired_wilcoxon(
    before: Sequence[float],
    after: Sequence[float],
    *,
    alternative: str = "greater",
) -> PairedResult:
    """Wilcoxon signed-rank on paired observations, with a rank-biserial effect size.

    For designs where the same units are measured twice — here, the same bats
    scored under a trained model and under its untrained control. The pairing is
    the whole point: between-bat variation in how much saliency an eye attracts
    is large and irrelevant, and differencing within a bat removes it.

    Non-parametric because n is 12-16 identities and the per-identity densities
    are right-skewed (a few images whose hot spot lands in the region pull the
    mean up), so a t-test's normality assumption is not comfortable here.

    ``alternative`` gives the one-sided p in the stated direction; the two-sided
    p is always returned alongside, so a directional hypothesis cannot quietly
    become a fishing expedition.

    The effect size is the matched-pairs rank-biserial correlation
    (Kerby 2014): (positive rank sum - negative rank sum) / total rank sum, in
    [-1, +1]. +1 means every pair moved the same way. It is reported because a
    p-value alone cannot say whether a difference matters.
    """
    x = np.asarray(before, dtype=float)
    y = np.asarray(after, dtype=float)
    if x.size != y.size:
        raise ValueError(f"paired inputs must be the same length; got {x.size} and {y.size}")
    if x.size < 2:
        raise ValueError("need at least 2 pairs")

    diff = y - x
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        # Every pair identical: no evidence of change, and scipy would raise.
        return PairedResult(
            n_pairs=int(x.size),
            median_before=float(np.median(x)),
            median_after=float(np.median(y)),
            median_difference=0.0,
            wilcoxon_w=0.0,
            p_value=1.0,
            p_value_one_sided=1.0,
            matched_pairs_rank_biserial=0.0,
            effect_magnitude="negligible",
        )

    two_sided = stats.wilcoxon(y, x, alternative="two-sided", zero_method="wilcox")
    one_sided = stats.wilcoxon(y, x, alternative=alternative, zero_method="wilcox")

    ranks = stats.rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    total = pos + neg
    rbc = (pos - neg) / total if total > 0 else 0.0

    return PairedResult(
        n_pairs=int(x.size),
        median_before=float(np.median(x)),
        median_after=float(np.median(y)),
        median_difference=float(np.median(diff)),
        wilcoxon_w=float(two_sided.statistic),
        p_value=float(two_sided.pvalue),
        p_value_one_sided=float(one_sided.pvalue),
        matched_pairs_rank_biserial=float(rbc),
        effect_magnitude=interpret_delta(rbc),
    )


def compare_groups(a: Sequence[float], b: Sequence[float]) -> ComparisonResult:
    """Mann-Whitney U plus both effect sizes for two independent groups.

    Independent, not paired: comparing two species means comparing two disjoint
    sets of animals, so there is nothing to pair on.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
    delta = cliffs_delta(x, y)
    return ComparisonResult(
        n_a=int(x.size),
        n_b=int(y.size),
        median_a=float(np.median(x)),
        median_b=float(np.median(y)),
        mann_whitney_u=float(u_stat),
        p_value=float(p_value),
        cliffs_delta=delta,
        delta_magnitude=interpret_delta(delta),
        hedges_g=hedges_g(x, y),
    )


# ---------------------------------------------------------------------------
# Equivalence testing
# ---------------------------------------------------------------------------


def tost(
    a: Sequence[float],
    b: Sequence[float],
    *,
    margin: float,
    margin_in_g_units: bool = True,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two one-sided tests for equivalence of two independent means.

    Standard null-hypothesis testing can only fail to find a difference; it can
    never support equivalence. TOST inverts the logic: the null is
    *non-equivalence*, so a significant result actively supports "these two are
    the same within the declared margin". The margin is a scientific judgement
    and must be stated with the result, never chosen after seeing the data.

    Args:
        a, b: Independent samples.
        margin: Half-width of the equivalence interval. Interpreted in pooled
            standard-deviation units when *margin_in_g_units* is true (so 0.5
            means "differences under half a between-unit SD do not matter"),
            otherwise in the raw units of the data.
        margin_in_g_units: See *margin*.
        alpha: Level for each one-sided test.

    Returns:
        :class:`EquivalenceResult`; ``equivalent`` is ``p_tost < alpha``.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        raise ValueError("TOST needs at least 2 observations per group")

    if margin_in_g_units:
        pooled_var = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
        raw_margin = float(margin * np.sqrt(pooled_var))
        units = f"{margin} pooled SD ({raw_margin:.6g} raw)"
    else:
        raw_margin = float(margin)
        units = f"{margin:.6g} raw"

    mean_diff = float(x.mean() - y.mean())
    # Welch standard error and degrees of freedom — group sizes and variances
    # both differ here (16 vs 12 identities), so pooled-variance t is wrong.
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    se = float(np.sqrt(v1 + v2))
    if se <= 0:
        raise ValueError("zero standard error — samples are constant")
    df = float((v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)))

    low, high = -raw_margin, raw_margin
    # H0_lower: diff <= low  (reject when diff is comfortably above low)
    p_lower = float(stats.t.sf((mean_diff - low) / se, df))
    # H0_upper: diff >= high (reject when diff is comfortably below high)
    p_upper = float(stats.t.cdf((mean_diff - high) / se, df))
    p_tost = max(p_lower, p_upper)

    # The (1 - 2*alpha) CI is the interval TOST is equivalent to inspecting.
    t_crit = float(stats.t.ppf(1.0 - alpha, df))
    return EquivalenceResult(
        mean_difference=mean_diff,
        margin_low=low,
        margin_high=high,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        ci_low=mean_diff - t_crit * se,
        ci_high=mean_diff + t_crit * se,
        equivalent=bool(p_tost < alpha),
        margin_units=units,
    )


# ---------------------------------------------------------------------------
# Correlated-fold t-test
# ---------------------------------------------------------------------------


def corrected_resampled_ttest(
    differences: Sequence[float],
    *,
    n_train: int,
    n_test: int,
) -> TTestResult:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    Resampled train/test splits overlap, so per-fold score differences are
    positively correlated and the naive variance ``s^2 / n`` is too small. The
    correction replaces it with ``(1/n + n_test/n_train) * s^2``, inflating the
    variance by the test-to-train size ratio.

    Use for *paired* comparisons that share folds — e.g. ArcFace vs AdaFace
    evaluated on the same 20 partitions. It does **not** apply across species,
    which cannot share folds; use :func:`hierarchical_bootstrap` there.

    Args:
        differences: Per-fold paired score differences.
        n_train: Training-set size in one fold.
        n_test: Test-set size in one fold.
    """
    d = np.asarray(differences, dtype=float)
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 folds")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")

    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    # Guard on a *relative* threshold, not on exact zero: identical inputs
    # still leave float-rounding noise behind (var([0.1]*3) is ~3e-34, not 0),
    # which would otherwise sail through and produce a t-statistic of ~1e16.
    scale = max(abs(mean_d), float(np.max(np.abs(d))), 1e-300)
    if not np.isfinite(var_d) or np.sqrt(var_d) <= 1e-12 * scale:
        raise ValueError("zero variance across folds — every fold scored identically")

    corrected_var = (1.0 / n + n_test / n_train) * var_d
    t_stat = mean_d / float(np.sqrt(corrected_var))
    df = float(n - 1)
    return TTestResult(
        mean_difference=mean_d,
        t_statistic=float(t_stat),
        df=df,
        p_value=float(2.0 * stats.t.sf(abs(t_stat), df)),
        correction=f"nadeau-bengio (n={n}, n_test/n_train={n_test / n_train:.4f})",
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for a one-sample statistic."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    draws = np.array(
        [statistic(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
    # Two-sided p for "statistic differs from zero", from the bootstrap draws.
    frac_le = float(np.mean(draws <= 0.0))
    frac_ge = float(np.mean(draws >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return BootstrapResult(
        estimate=float(statistic(x)),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=p,
        n_boot=n_boot,
        confidence=confidence,
    )


def hierarchical_bootstrap(
    group_a: Mapping[str, Sequence[float]],
    group_b: Mapping[str, Sequence[float]],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Two-level bootstrap of the difference between two clustered groups.

    Resamples **identities** with replacement, then observations within each
    drawn identity. This is the honest inference unit for this project: images
    within one bat are near-duplicates from the same video, so treating them as
    independent replicates fabricates precision.

    Args:
        group_a, group_b: identity -> observations, one mapping per group.
        statistic: Applied to the pooled observations of each resampled group.
        n_boot: Bootstrap replicates.
        confidence: CI level.
        seed: RNG seed.

    Returns:
        :class:`BootstrapResult` for ``statistic(a) - statistic(b)``.
    """
    ids_a = [k for k, v in group_a.items() if len(v) > 0]
    ids_b = [k for k, v in group_b.items() if len(v) > 0]
    if not ids_a or not ids_b:
        raise ValueError("both groups need at least one non-empty identity")

    arrays_a = {k: np.asarray(group_a[k], dtype=float) for k in ids_a}
    arrays_b = {k: np.asarray(group_b[k], dtype=float) for k in ids_b}
    rng = np.random.default_rng(seed)

    def resample(ids: list[str], arrays: dict[str, np.ndarray]) -> float:
        drawn = rng.choice(np.asarray(ids, dtype=object), size=len(ids), replace=True)
        pooled = [
            rng.choice(arrays[str(identity)], size=arrays[str(identity)].size, replace=True)
            for identity in drawn
        ]
        return statistic(np.concatenate(pooled))

    draws = np.array(
        [resample(ids_a, arrays_a) - resample(ids_b, arrays_b) for _ in range(n_boot)],
        dtype=float,
    )
    observed = statistic(np.concatenate([arrays_a[k] for k in ids_a])) - statistic(
        np.concatenate([arrays_b[k] for k in ids_b])
    )

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
    frac_le = float(np.mean(draws <= 0.0))
    frac_ge = float(np.mean(draws >= 0.0))
    return BootstrapResult(
        estimate=float(observed),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=min(1.0, 2.0 * min(frac_le, frac_ge)),
        n_boot=n_boot,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# p-value aggregation and multiplicity
# ---------------------------------------------------------------------------


def permutation_p(n_at_least_as_extreme: int, n_permutations: int) -> float:
    """Monte-Carlo permutation p-value ``(Z + 1) / (B + 1)``.

    The ``+1`` on both sides is not cosmetic: it keeps the test valid and stops
    a p-value from ever being reported as exactly zero. Note the resolution
    floor — with ``B = 100`` no result can be smaller than ``1/101 = 0.0099``,
    so any claim below that needs a larger ``B``.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if not 0 <= n_at_least_as_extreme <= n_permutations:
        raise ValueError("n_at_least_as_extreme must be in [0, n_permutations]")
    return (n_at_least_as_extreme + 1) / (n_permutations + 1)


def permutation_p_floor(n_permutations: int) -> float:
    """Smallest p-value ``n_permutations`` can produce."""
    return 1.0 / (n_permutations + 1)


def harmonic_mean_p(
    p_values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    n_total: int | None = None,
) -> float:
    """Asymptotically exact harmonic-mean p-value (Wilson 2019, PNAS).

    The right combiner when the individual tests are **dependent** — which
    cross-validation folds are, since they share training data. Fisher's method
    assumes independence and is anti-conservative here.

    Implements the ``harmonicmeanp::p.hmp`` formula: with ``HMP = sum(w) /
    sum(w / p)``, the p-value is ``sum(w) * P(X > sum(w) / HMP)`` where ``X``
    follows a Landau distribution with location ``log(L) + 0.874`` and scale
    ``pi / 2``. Verified against the package's published worked example
    (L = 6524432, HMP = 8.734522e-4 -> p = 1.343897e-3).

    Args:
        p_values: Individual p-values in (0, 1].
        weights: Optional weights; default ``1 / n_total`` each. Must sum to
            at most 1.
        n_total: ``L``, the number of tests in the *whole* family the
            multiple-testing claim covers. Defaults to ``len(p_values)``; pass
            the larger family size when combining a subset.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("no p-values given")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0, 1]")

    n = int(n_total) if n_total is not None else p.size
    if n < p.size:
        raise ValueError("n_total must be >= len(p_values)")

    w = np.full(p.size, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    if w_sum > 1.0 + 1e-6:
        raise ValueError(f"weights must sum to at most 1; got {w_sum}")

    hmp = w_sum / float(np.sum(w / p))
    location = float(np.log(n) + 1.0 + digamma(1.0) - np.log(2.0 / np.pi))
    tail = float(stats.landau.sf(w_sum / hmp, loc=location, scale=np.pi / 2.0))
    return float(min(1.0, w_sum * tail))


def combine_pvalues(
    p_values: Sequence[float],
    *,
    method: Literal["hmp", "fisher", "stouffer"] = "hmp",
    n_total: int | None = None,
) -> float:
    """Combine p-values across folds or cells.

    ``hmp`` (default) tolerates dependence between tests; ``fisher`` and
    ``stouffer`` assume independence and will be anti-conservative on
    overlapping cross-validation folds.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("no p-values given")
    if p.size == 1:
        return float(p[0])
    if method == "hmp":
        return harmonic_mean_p(p, n_total=n_total)
    if method in ("fisher", "stouffer"):
        return float(stats.combine_pvalues(p, method=method).pvalue)
    raise ValueError(f"unknown method: {method!r}")


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), input order preserved.

    Use across the species x model x background cells; without it, 18
    simultaneous tests at alpha = 0.05 expect roughly one false positive.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    return np.asarray(stats.false_discovery_control(p, method="bh"), dtype=float)
