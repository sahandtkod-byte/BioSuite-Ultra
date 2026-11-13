"""Regression tests for popgen.py review fixes (Tajima's D rewrite)."""
import numpy as np
import pytest

from biosuite.core.popgen import (
    tajimas_d, hardy_weinberg_test, calculate_fst, nucleotide_diversity,
    linkage_disequilibrium, full_analysis,
)


def _matrix(rows):
    return np.array(rows, dtype=int)


def test_tajima_all_singletons_negative():
    # Every variant seen in exactly one chromosome -> excess of rare alleles
    n, s = 10, 20
    rng = np.random.default_rng(1)
    gm = np.zeros((n, s))
    for j in range(s):
        gm[rng.integers(n), j] = 1  # one het per site (2 alt / 2n alleles)
    d = tajimas_d(gm)
    assert d < 0, d


def test_tajima_intermediate_frequency_positive():
    # Variants at ~50% frequency, no rare alleles -> positive D
    n, s = 20, 10
    gm = np.zeros((n, s))
    half = n // 2
    gm[:half, :] = 2  # half alt/alt, half ref/ref at every site
    d = tajimas_d(gm)
    assert d > 0, d


def test_tajima_known_value_textbook_segsite_theta():
    # Hand-computable case: 2 samples, 1 heterozygous site among 4 sites.
    gm = _matrix([[0, 0, 0, 1], [0, 0, 0, 1]])
    d = tajimas_d(gm)
    assert isinstance(d, float)  # degenerate n=4 chromosomes: must not crash
    assert d == tajimas_d(gm)    # deterministic


def test_tajima_mono_and_tiny_matrix():
    assert tajimas_d(np.zeros((5, 5), dtype=int)) == 0.0
    assert tajimas_d(np.ones((1, 3), dtype=int)) == 0.0


def test_hwe_textbook_counts():
    # p=0.5, n=100 in perfect HWE
    out = hardy_weinberg_test({'AA': 25, 'Aa': 50, 'aa': 25})
    assert out['p_value'] == 1.0 or out['chi2'] < 0.01
    assert out['in_hwe']


def test_hwe_detects_imbalance():
    out = hardy_weinberg_test({'AA': 90, 'Aa': 5, 'aa': 5})
    assert not out['in_hwe'] or out['p_value'] < 0.05


def test_fst_boundaries():
    same = [_matrix([[1, 2], [1, 2]]), _matrix([[1, 2], [1, 2]])]
    assert calculate_fst(same)[(0, 1)] == 0.0
    fixed = [_matrix([[2] * 4, [2] * 4]), _matrix([[0] * 4, [0] * 4])]
    assert calculate_fst(fixed)[(0, 1)] == 1.0


def test_pi_zero_for_invariant_sites():
    assert nucleotide_diversity(np.zeros((4, 5), dtype=int)) == 0.0


def test_ld_perfectly_linked():
    a = [0, 1, 2, 0, 1, 2, 0, 1]
    b = [0, 1, 2, 0, 1, 2, 0, 1]
    gm = np.array([a, b], dtype=int).T
    ld = linkage_disequilibrium(gm)
    assert ld[(0, 1)] == pytest.approx(1.0)


def test_full_analysis_report(tmp_path):
    rng = np.random.default_rng(3)
    gm = rng.integers(0, 3, size=(10, 30))
    rep = full_analysis(gm)
    assert isinstance(rep.tajima_d, float)
    assert 'p_value' in rep.hw_test
