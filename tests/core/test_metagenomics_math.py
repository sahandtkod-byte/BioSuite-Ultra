"""Metagenomics diversity math + builtin classifier smoke tests."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import metagenomics as mg


def test_shannon_entropy_uniform_vs_peaked():
    uniform = mg.shannon_entropy(np.array([50, 50, 50, 50]))
    peaked = mg.shannon_entropy(np.array([97, 1, 1, 1]))
    assert uniform > peaked
    assert uniform == pytest.approx(np.log(4), rel=1e-3)


def test_simpson_index_bounds():
    uniform = mg.simpson_index(np.array([50, 50]))
    peaked = mg.simpson_index(np.array([99, 1]))
    assert peaked < uniform <= 1.0


def test_chao1_monotonic_with_singletons():
    counts_no_singles = np.array([5, 5, 5, 5])
    counts_with_singles = np.array([5, 5, 1, 1, 1, 1])
    assert mg.chao1_estimator(counts_with_singles) > mg.chao1_estimator(counts_no_singles)


def test_bray_curtis_bounds_and_symmetry():
    a = np.array([10.0, 0.0, 5.0])
    b = np.array([0.0, 10.0, 5.0])
    d = mg.bray_curtis_distance(a, b)
    assert 0 <= d <= 1
    assert mg.bray_curtis_distance(a, a) == 0
    assert abs(d - mg.bray_curtis_distance(b, a)) < 1e-12


def test_alpha_diversity_from_table():
    table = pd.DataFrame({'taxon': ['A', 'B', 'C', 'D'], 'count': [40, 30, 20, 10]})
    out = mg.compute_alpha_diversity(table)
    assert 'shannon' in out and 'simpson' in out and 'chao1' in out


def test_beta_diversity_aligns_taxa():
    t1 = pd.DataFrame({'taxon': ['A', 'B'], 'count': [10, 5]})
    t2 = pd.DataFrame({'taxon': ['B', 'C'], 'count': [5, 10]})
    d = mg.compute_beta_diversity([t1, t2])
    assert d.shape == (2, 2)
    assert d[0, 0] == 0 and d[0, 1] > 0
    assert np.allclose(d, d.T)


def test_builtin_classifier_smoke(tmp_path):
    fa = tmp_path / 'reads.fasta'
    seqs = ['ACGT' * 50 + 'GCGCGCGC']
    fa.write_text('\n'.join(f'>r{i}\n{s}' for i, s in enumerate(seqs)))
    with pytest.warns(Warning):
        res = mg.classify_reads(str(fa))
    assert res is not None


def test_classify_reads_missing_file():
    res = mg.classify_reads('/nonexistent/reads.fasta')
    assert res.engine == 'none'


def test_format_metagenomics_report(tmp_path):
    fa = tmp_path / 'reads.fasta'
    fa.write_text('>r0\n' + 'ACGT' * 25)
    with pytest.warns(Warning):
        res = mg.classify_reads(str(fa))
    txt = mg.format_metagenomics_report(res)
    assert isinstance(txt, str)
