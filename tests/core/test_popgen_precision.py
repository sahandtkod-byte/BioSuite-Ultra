"""Numerical-precision regression tests for popgen (NEW-19).

``hardy_weinberg_test`` rounded its return values - ``p_value`` to six decimal
places - so every p below 5e-7 was reported as exactly 0.0.  That is not a
valid p-value for a continuous distribution, it makes ``-log10(p)`` infinite
(which is what Manhattan plots and FDR corrections consume), and it silently
undid the earlier switch from ``1 - chi2.cdf`` to ``chi2.sf``.

Rounding now happens at display time in ``format_popgen_report``.
"""
import math
import random

import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from biosuite.core.popgen import hardy_weinberg_test


# ── the p-value is no longer destroyed by rounding ──────────────────────────

def test_a_small_p_value_survives():
    """This case returned exactly 0.0 before."""
    result = hardy_weinberg_test({"AA": 60, "Aa": 10, "aa": 60})
    assert result["p_value"] > 0.0
    assert result["p_value"] == pytest.approx(5.030e-22, rel=1e-3)


def test_minus_log10_p_is_finite_for_a_strong_deviation():
    result = hardy_weinberg_test({"AA": 60, "Aa": 10, "aa": 60})
    assert math.isfinite(-math.log10(result["p_value"]))


def test_p_value_is_not_rounded_to_six_places():
    result = hardy_weinberg_test({"AA": 100, "Aa": 30, "aa": 100})
    assert 0.0 < result["p_value"] < 5e-7
    assert round(result["p_value"], 6) == 0.0, "test case no longer exercises the bug"


def test_chi2_is_not_rounded_to_four_places():
    result = hardy_weinberg_test({"AA": 68, "Aa": 291, "aa": 433})
    assert result["chi2"] != round(result["chi2"], 4)


# ── independent oracle ──────────────────────────────────────────────────────

@pytest.mark.parametrize("counts", [
    {"AA": 25, "Aa": 50, "aa": 25},
    {"AA": 68, "Aa": 291, "aa": 433},
    {"AA": 369, "Aa": 410, "aa": 117},
    {"AA": 90, "Aa": 10, "aa": 0},
    {"AA": 1, "Aa": 1, "aa": 1},
])
def test_matches_an_independently_computed_chi_square(counts):
    result = hardy_weinberg_test(counts)
    n_aa, n_ab, n_bb = counts["AA"], counts["Aa"], counts["aa"]
    n = n_aa + n_ab + n_bb
    p = (2 * n_aa + n_ab) / (2 * n)
    q = 1 - p
    expected = [n * p * p, 2 * n * p * q, n * q * q]
    observed = [n_aa, n_ab, n_bb]
    oracle = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    assert result["chi2"] == pytest.approx(oracle, rel=1e-12, abs=1e-12)
    assert result["p_value"] == pytest.approx(chi2_dist.sf(oracle, 1),
                                              rel=1e-12, abs=1e-300)


def test_randomised_differential_check_against_scipy():
    rng = random.Random(20260904)
    for _ in range(500):
        counts = {"AA": rng.randint(0, 500), "Aa": rng.randint(0, 500),
                  "aa": rng.randint(0, 500)}
        if sum(counts.values()) == 0:
            continue
        result = hardy_weinberg_test(counts)
        n = sum(counts.values())
        p = (2 * counts["AA"] + counts["Aa"]) / (2 * n)
        q = 1 - p
        expected = [n * p * p, 2 * n * p * q, n * q * q]
        observed = [counts["AA"], counts["Aa"], counts["aa"]]
        oracle = sum((o - e) ** 2 / e
                     for o, e in zip(observed, expected) if e > 0)
        assert result["chi2"] == pytest.approx(oracle, rel=1e-9, abs=1e-12), counts


# ── log10_p_value ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("counts", [
    {"AA": 68, "Aa": 291, "aa": 433},
    {"AA": 60, "Aa": 10, "aa": 60},
    {"AA": 100, "Aa": 30, "aa": 100},
])
def test_log10_p_agrees_with_log10_of_p_where_p_is_representable(counts):
    result = hardy_weinberg_test(counts)
    assert result["p_value"] > 0
    assert result["log10_p_value"] == pytest.approx(
        math.log10(result["p_value"]), rel=1e-9)


@pytest.mark.parametrize("counts", [
    {"AA": 5000, "Aa": 0, "aa": 5000},
    {"AA": 500000, "Aa": 0, "aa": 500000},
])
def test_log10_p_stays_finite_where_p_underflows(counts):
    """The whole point: genome-scale results must remain usable."""
    result = hardy_weinberg_test(counts)
    assert result["p_value"] == 0.0, "test case no longer exercises underflow"
    assert math.isfinite(result["log10_p_value"])
    assert result["log10_p_value"] < -1000


def test_the_identity_used_for_log10_p_matches_scipy_where_scipy_is_finite():
    """sf_chi2(x; df=1) == 2 * sf_normal(sqrt(x)) is an identity, not a fit."""
    for x in (0.5, 1.0, 10.0, 100.0, 500.0, 1000.0, 1400.0):
        scipy_log = chi2_dist.logsf(x, 1)
        assert math.isfinite(scipy_log), "scipy unexpectedly underflowed here"
        identity = math.log(2.0) + norm.logsf(math.sqrt(x))
        assert identity == pytest.approx(scipy_log, rel=1e-9, abs=1e-9)


def test_log10_p_is_zero_for_a_perfect_fit():
    result = hardy_weinberg_test({"AA": 25, "Aa": 50, "aa": 25})
    assert result["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["log10_p_value"] == pytest.approx(0.0, abs=1e-12)


def test_log10_p_is_monotonic_in_chi2():
    previous = 1.0
    for het in (50, 40, 30, 20, 10, 5, 1, 0):
        result = hardy_weinberg_test({"AA": 25, "Aa": het, "aa": 25})
        assert result["log10_p_value"] <= previous + 1e-12
        previous = result["log10_p_value"]


# ── display formatting still rounds ─────────────────────────────────────────

def test_the_report_formats_small_p_values_in_scientific_notation():
    from biosuite.core.popgen import PopGenReport, format_popgen_report
    report = PopGenReport()
    report.hw_test = hardy_weinberg_test({"AA": 60, "Aa": 10, "aa": 60})
    text = format_popgen_report(report)
    assert "p=" in text
    assert "p=0.0000" not in text, "a tiny p must not be displayed as zero"
    assert "e-" in text
