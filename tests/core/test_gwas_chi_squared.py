"""One SNP, four counts: what the 2x2 test accepts and what it answers.

Three of these numbers reach the printed report (`odds_ratio`, `p_value`,
`chi2_stat`), and the counts arrive from a file the user pointed at, so a table
the test cannot answer must say so instead of inventing a ratio or dying inside
scipy with a message about "expected frequencies".
"""
import math

import numpy as np
import pytest

from biosuite.core.gwas import gwas_chi_squared


def or_of(**kwargs):
    return gwas_chi_squared(**kwargs)["odds_ratio"]


# ── the direction and the value of the odds ratio ───────────────────────────

def test_an_alt_allele_enriched_in_cases_has_an_odds_ratio_above_one():
    """The formula used to be the reciprocal, which inverted every result."""
    res = gwas_chi_squared(controls_alt_count=100, cases_alt_count=200,
                           controls_total=1000, cases_total=1000)

    assert res["odds_ratio"] == pytest.approx(2.25), res
    assert (200 * 900) / (800 * 100) == pytest.approx(2.25)


def test_an_alt_allele_depleted_in_cases_has_an_odds_ratio_below_one():
    res = gwas_chi_squared(controls_alt_count=200, cases_alt_count=100,
                           controls_total=1000, cases_total=1000)

    assert res["odds_ratio"] == pytest.approx(1 / 2.25), res


def test_a_protective_allele_absent_from_cases_scores_exactly_zero():
    assert or_of(controls_alt_count=50, cases_alt_count=0,
                 controls_total=100, cases_total=100) == 0.0


def test_an_empty_control_cell_is_corrected_instead_of_fabricating_a_number():
    """`max(denominator, 1)` answered the numerator itself: OR=5000.0 here.

    A ratio that grows with the sample size is not a ratio, and 5000 was printed
    in the report exactly like the honest ones.  Haldane-Anscombe adds half an
    observation to each cell, which keeps the estimate finite and shrinks it
    towards 1 as the evidence allows.
    """
    corrected = ((50 + 0.5) * (100 + 0.5)) / ((50 + 0.5) * (0 + 0.5))

    assert or_of(controls_alt_count=0, cases_alt_count=50,
                 controls_total=100, cases_total=100) == pytest.approx(corrected)
    assert or_of(controls_alt_count=0, cases_alt_count=50,
                 controls_total=100, cases_total=100) == pytest.approx(201.0)


def test_a_perfect_separator_is_reported_as_a_ratio_not_as_its_own_numerator():
    """The old code answered 5000.0 for the first case here — 50*100, no denominator.

    A corrected estimate for a table with an empty cell still grows with the
    sample, which is honest: the odds ratio it is approximating is unbounded.
    What must not happen is the number silently becoming the numerator.
    """
    small = gwas_chi_squared(0, 50, 100, 100)["odds_ratio"]
    large = gwas_chi_squared(0, 500, 1000, 1000)["odds_ratio"]

    assert small == pytest.approx(201.0), small
    assert large == pytest.approx(2001.0), large
    assert large > small


def test_a_non_degenerate_table_is_untouched_by_the_correction():
    """The 0.5 is only added when a cell is empty, so pinned values stay exact."""
    assert or_of(controls_alt_count=10, cases_alt_count=90,
                 controls_total=100, cases_total=100) == pytest.approx(81.0)
    assert or_of(controls_alt_count=15, cases_alt_count=85,
                 controls_total=100, cases_total=100) == pytest.approx(32.1111, abs=1e-4)


# ── the test itself ─────────────────────────────────────────────────────────

def test_chi_squared_uses_yates_correction_on_the_same_table():
    counts = {"controls_alt_count": 10, "cases_alt_count": 90,
              "controls_total": 100, "cases_total": 100}
    res = gwas_chi_squared(**counts)
    observed = np.array([[10, 90], [90, 10]], dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()

    assert res["chi2_stat"] == pytest.approx(
        float((((abs(observed - expected) - 0.5) ** 2) / expected).sum()))


def test_a_perfectly_separating_snp_is_significant():
    res = gwas_chi_squared(controls_alt_count=1, cases_alt_count=99,
                           controls_total=100, cases_total=100)

    assert 0.0 <= res["p_value"] < 1e-20, res
    assert res["chi2_stat"] > 100, res


def test_identical_allele_frequencies_answer_a_null_result():
    res = gwas_chi_squared(controls_alt_count=50, cases_alt_count=50,
                           controls_total=100, cases_total=100)

    assert res["chi2_stat"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["odds_ratio"] == pytest.approx(1.0)


# ── counts that cannot be tested at all ─────────────────────────────────────

def test_an_absent_alt_allele_in_both_arms_is_refused_with_a_reason():
    """scipy's answer was ``expected frequencies has a zero element at (0, 0)``.

    That message names no SNP and no file, and one such row used to abort a whole
    run of a thousand SNPs.
    """
    with pytest.raises(ValueError, match="chi-squared test is undefined") as exc:
        gwas_chi_squared(controls_alt_count=0, cases_alt_count=0,
                         controls_total=100, cases_total=100)

    assert "absent in both arms" in str(exc.value), str(exc.value)
    assert "controls 0/100, cases 0/100" in str(exc.value), str(exc.value)


def test_an_alt_allele_fixed_in_both_arms_is_refused_too():
    with pytest.raises(ValueError, match="fixed in both arms"):
        gwas_chi_squared(controls_alt_count=100, cases_alt_count=100,
                         controls_total=100, cases_total=100)


@pytest.mark.parametrize("argument", [
    "controls_alt_count", "cases_alt_count", "controls_total", "cases_total"])
def test_a_negative_count_is_named_by_its_own_argument(argument):
    counts = {"controls_alt_count": 10, "cases_alt_count": 20,
              "controls_total": 100, "cases_total": 100}
    counts[argument] = -1 if argument.endswith("count") else -100

    with pytest.raises(ValueError, match=argument):
        gwas_chi_squared(**counts)


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), None, "abc", [5]])
def test_a_count_that_is_not_a_finite_number_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="controls_alt_count"):
        gwas_chi_squared(controls_alt_count=bad, cases_alt_count=20,
                         controls_total=100, cases_total=100)


def test_a_count_written_as_text_still_works_because_files_are_read_as_strings():
    assert gwas_chi_squared("10", "20", "100", "100")["odds_ratio"] > 0


@pytest.mark.parametrize("counts, expected", [
    ((150, 20, 100, 100), "controls_alt_count"),
    ((10, 200, 100, 100), "cases_alt_count"),
])
def test_an_allele_count_above_its_own_total_is_refused(counts, expected):
    """A negative reference count is what used to reach scipy instead."""
    with pytest.raises(ValueError, match=f"{expected} .* exceeds") as exc:
        gwas_chi_squared(controls_alt_count=counts[0], cases_alt_count=counts[1],
                         controls_total=counts[2], cases_total=counts[3])

    assert "exceeds" in str(exc.value), str(exc.value)


def test_a_zero_total_in_one_arm_is_a_refused_undefined_test():
    with pytest.raises(ValueError, match="chi-squared test is undefined"):
        gwas_chi_squared(controls_alt_count=0, cases_alt_count=50,
                         controls_total=0, cases_total=100)


def test_the_answer_carries_exactly_the_three_documented_keys():
    assert set(gwas_chi_squared(10, 20, 100, 100)) == {"chi2_stat", "p_value",
                                                       "odds_ratio"}
    assert math.isfinite(gwas_chi_squared(10, 20, 100, 100)["chi2_stat"])
