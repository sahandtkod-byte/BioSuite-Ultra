"""Population genetics: degenerate inputs, the constant-site skip and PCA.

`hardy_weinberg_test` already refuses a negative or non-numeric count, but a NaN
or an infinity passed the two checks and reached the chi-square arithmetic.
`nucleotide_diversity`, `tajimas_d` and `linkage_disequilibrium` each have an
early answer for the cases where the statistics are not defined — one sample, a
monomorphic alignment, a site nobody varies at — and none of those answers had
ever been produced in a test.  PCA is exercised both ways: with scikit-learn
present, and with the import blocked to prove the message the user gets instead.
"""
import sys

import numpy as np
import pytest

from biosuite.core import popgen as pg


# ── Hardy-Weinberg input validation ─────────────────────────────────────────

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_genotype_count_is_refused(value):
    with pytest.raises(ValueError, match="must be finite"):
        pg.hardy_weinberg_test({"AA": value, "Aa": 10, "aa": 10})


@pytest.mark.parametrize("key", ["AA", "Aa", "aa"])
def test_every_genotype_class_is_checked(key):
    counts = {"AA": 10, "Aa": 10, "aa": 10}
    counts[key] = float("nan")
    with pytest.raises(ValueError, match=f"genotype count {key!r} must be finite"):
        pg.hardy_weinberg_test(counts)


def test_a_numpy_scalar_count_is_accepted():
    """np.int64 is what a pandas value_counts() hands over; the guard lists it as a
    valid number, so it must not be turned into an error."""
    res = pg.hardy_weinberg_test({"AA": np.int64(25), "Aa": np.int64(50),
                                  "aa": np.int64(25)})
    assert res["chi2"] == pytest.approx(0.0, abs=1e-9)
    assert res["allele_freq_p"] == pytest.approx(0.5)


# ── statistics that are not defined on thin data ────────────────────────────

def test_nucleotide_diversity_needs_two_samples():
    assert pg.nucleotide_diversity(np.array([[0.0, 1.0, 2.0]])) == 0.0
    assert pg.nucleotide_diversity(np.zeros((0, 3))) == 0.0


def test_a_monomorphic_alignment_has_no_diversity_and_no_tajima_d():
    mono = np.zeros((6, 4), dtype=float)
    assert pg.nucleotide_diversity(mono) == 0.0
    assert pg.tajimas_d(mono) == 0.0, "var_d collapses to zero with no segregating sites"


def test_tajimas_d_on_a_single_sample():
    assert pg.tajimas_d(np.array([[0.0, 1.0, 2.0]])) == 0.0


def test_tajimas_d_stays_a_finite_number_on_random_sites():
    """The value depends on the seed, so what is pinned is that the estimator
    returns a number at all — the degenerate branches above return 0.0 and a
    silent collapse to zero would otherwise look the same."""
    rng = np.random.default_rng(2)
    sites = rng.integers(0, 3, size=(40, 6)).astype(float)
    d = pg.tajimas_d(sites)
    assert isinstance(d, float)
    assert np.isfinite(d)
    assert d != 0.0
    assert -10.0 < d < 10.0


# ── linkage disequilibrium ──────────────────────────────────────────────────

@pytest.fixture()
def three_sites():
    m = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 0],
                  [0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=float)
    return m


def test_a_constant_site_is_skipped_rather_than_dividing_by_zero(three_sites):
    three_sites[:, 0] = 1.0          # nobody varies here
    ld = pg.linkage_disequilibrium(three_sites)
    assert sorted(ld) == [(1, 2)]
    assert all(0 not in pair for pair in ld)


def test_perfectly_linked_sites_score_r2_one():
    a = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    m = np.column_stack([a, a])
    ld = pg.linkage_disequilibrium(m)
    assert ld == {(0, 1): 1.0}


def test_r_squared_ignores_the_direction_of_the_relationship():
    """A site that rises as the other falls scores the same as one that rises
    with it: LD here is r squared, not r, so the sign must not matter."""
    up = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    same = pg.linkage_disequilibrium(np.column_stack([up, up]))
    inverted = pg.linkage_disequilibrium(np.column_stack([up, up[::-1]]))
    assert same == {(0, 1): 1.0}
    assert inverted == {(0, 1): 1.0}


def test_a_partially_related_pair_scores_the_measured_fraction(three_sites):
    """These three sites are a Latin square, so the pairwise r squared values are
    exactly the numbers pinned below — a change in the pairing or the squaring
    shows up immediately."""
    ld = pg.linkage_disequilibrium(three_sites)
    assert ld == {(0, 1): 0.0625, (0, 2): 0.25, (1, 2): 0.25}


def test_max_pairs_thins_the_pair_list_rather_than_truncating_a_full_scan(three_sites):
    """max_pairs drives a sampling step (n_sites // sqrt(max_pairs)) *before* the
    window loop, so a small cap can skip site pairs that exist — with three sites
    and max_pairs=3 the pair (1, 2) is never visited.  Pinned because a caller
    reading "maximum number of site pairs" would expect all three."""
    assert list(pg.linkage_disequilibrium(three_sites, max_pairs=1)) == [(0, 1)]
    assert list(pg.linkage_disequilibrium(three_sites, max_pairs=2)) == [(0, 1), (0, 2)]
    assert list(pg.linkage_disequilibrium(three_sites, max_pairs=3)) == [(0, 1), (0, 2)]
    assert sorted(pg.linkage_disequilibrium(three_sites)) == [(0, 1), (0, 2), (1, 2)]


def test_every_site_pair_within_the_window_is_computed_at_the_default_cap():
    """The 20-site window is the real limit on a wide matrix, not max_pairs: with
    the default every adjacent pair of a 6-site alignment shows up."""
    rng = np.random.default_rng(6)
    m = rng.integers(0, 3, size=(12, 6)).astype(float)
    ld = pg.linkage_disequilibrium(m)
    assert all(j > i for i, j in ld)
    assert {(0, 1), (0, 5), (4, 5)} <= set(ld), "near and far pairs both counted"
    assert len(ld) <= 6 * 5 // 2


def test_a_single_site_matrix_has_no_pairs():
    assert pg.linkage_disequilibrium(np.array([[0.0], [1.0], [2.0]])) == {}


# ── PCA ─────────────────────────────────────────────────────────────────────

def test_pca_returns_coordinates_and_variance_explained(three_sites):
    coords, explained = pg.pca_genotypes(three_sites)
    assert coords.shape == (6, 2)
    assert explained.shape == (2,)
    # The matrix has three independent sites, so two components cannot cover all
    # of the variance; the split below is what sklearn reports for it.
    assert explained == pytest.approx([0.614357, 0.25], abs=1e-6)
    assert explained.sum() < 1.0
    # Centring is the implementation's own step, so the mean must be gone.
    assert np.allclose(coords.mean(axis=0), 0.0, atol=1e-9)


def test_pca_honours_the_component_count(three_sites):
    coords, explained = pg.pca_genotypes(three_sites, n_components=1)
    assert coords.shape == (6, 1)
    assert explained.shape == (1,)
    assert explained[0] == pytest.approx(0.614357, abs=1e-6)


def test_pca_reports_a_useful_message_when_sklearn_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    with pytest.raises(ImportError, match="scikit-learn is required for PCA"):
        pg.pca_genotypes(np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]))


# ── the whole pipeline and its report ───────────────────────────────────────

def test_full_analysis_reports_every_statistic_it_computed(three_sites):
    report = pg.full_analysis(three_sites)
    assert report.num_sites == 3
    assert report.num_populations == 1
    assert report.nucleotide_diversity == pytest.approx(0.6)
    assert isinstance(report.tajima_d, float)
    assert report.hw_test["allele_freq_p"] == pytest.approx(0.5)
    text = pg.format_popgen_report(report)
    assert "Sites: 3" in text
    assert f"Nucleotide diversity (pi): {report.nucleotide_diversity:.6f}" in text
    assert "HWE (site 0):" in text
    assert "In HWE:" in text


def test_report_omits_the_blocks_that_were_not_computed():
    report = pg.PopGenReport(num_sites=2, num_populations=1, fst={},
                             nucleotide_diversity=0.123456, tajima_d=-1.2345,
                             hw_test=None, ld={}, message="")
    text = pg.format_popgen_report(report)
    assert "HWE" not in text
    assert "Avg LD" not in text
    assert "Note:" not in text
    assert "Nucleotide diversity (pi): 0.123456" in text
    assert "Tajima's D: -1.2345" in text


def test_a_tiny_p_value_is_printed_in_scientific_notation():
    """Rounding the stored value would have shown p=0.0000 and read as exactly
    zero; the report formats instead."""
    report = pg.PopGenReport(num_sites=1, num_populations=1, fst={},
                             nucleotide_diversity=0.0, tajima_d=0.0,
                             hw_test={"chi2": 40.0, "p_value": 2.75e-10},
                             ld={(0, 1): 0.5}, message="some note")
    text = pg.format_popgen_report(report)
    assert "p=2.750e-10" in text
    assert "p=0.0000" not in text
    assert "In HWE: No" in text
    assert "Avg LD (r2): 0.5000 across 1 pairs" in text
    assert text.endswith("Note: some note")


def test_full_analysis_splits_populations_when_labels_are_given():
    """The labels used to change only the population count while `fst` stayed an
    empty dict; the statistic that needs them is now computed, by grouping the
    rows the same way `calculate_fst` expects."""
    m = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 2], [1, 2]], dtype=float)
    labels = np.array(["x", "x", "x", "y", "y", "y"])
    report = pg.full_analysis(m, population_labels=labels)
    assert report.num_populations == 2
    assert list(report.fst) == [(0, 1)]
    assert 0.0 < report.fst[(0, 1)] <= 1.0
    assert report.fst == pg.calculate_fst([m[labels == "x"], m[labels == "y"]])
    text = pg.format_popgen_report(report)
    assert "Populations: 2" in text
    assert text.count("FST (Nei's Gst):") == 1
    assert f"FST (Nei's Gst): {report.fst[(0, 1)]:.4f} across 1 pairs" in text


def test_full_analysis_without_labels_leaves_fst_empty(three_sites):
    """No grouping was supplied, so no pairwise statistic may be invented."""
    report = pg.full_analysis(three_sites)
    assert report.fst == {}
    assert "FST" not in pg.format_popgen_report(report)


def test_three_populations_score_every_pair():
    m = np.array([[0, 0], [1, 1], [2, 2], [0, 2], [1, 0], [2, 1]], dtype=float)
    labels = ["a", "a", "b", "b", "c", "c"]
    report = pg.full_analysis(m, population_labels=labels)
    assert report.num_populations == 3
    assert sorted(report.fst) == [(0, 1), (0, 2), (1, 2)]
    assert "across 3 pairs" in pg.format_popgen_report(report)
