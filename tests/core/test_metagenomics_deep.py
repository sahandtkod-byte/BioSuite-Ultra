"""Deep tests for core/metagenomics.py (classifier + diversity statistics)."""
import numpy as np
import pytest

from biosuite.core import metagenomics as mg


# ── alpha diversity math ─────────────────────────────────────────────────────

def test_shannon_entropy_known_values():
    assert mg.shannon_entropy([1, 1, 1, 1]) > mg.shannon_entropy([4, 0, 0, 0])
    assert mg.shannon_entropy([2, 2]) == pytest.approx(np.log(2), rel=1e-6)
    assert mg.shannon_entropy([5]) == pytest.approx(0.0)


def test_simpson_index():
    assert mg.simpson_index([1, 1, 1, 1]) > mg.simpson_index([4])
    assert mg.simpson_index([0, 0, 0]) in (0.0, 1.0)   # degenerate, guarded


def test_chao1_estimator():
    # 60 species once, 30 twice, rest zero -> finite positive estimate
    counts = [1] * 60 + [2] * 30
    est = mg.chao1_estimator(counts)
    assert np.isfinite(est) and est >= 0


def test_bray_curtis_known():
    # BC dissimilarity = sum|a-b| / sum(a+b): |20| / 40 = 0.5
    a = [10, 10, 0]
    b = [10, 0, 10]
    assert mg.bray_curtis_distance(a, b) == pytest.approx(0.5, abs=1e-9)
    assert mg.bray_curtis_distance(a, a) == pytest.approx(0.0)


def test_compute_alpha_diversity():
    import pandas as pd
    table = pd.DataFrame({'taxon': ['speciesA', 'speciesB'], 'count': [50, 50]})
    out = mg.compute_alpha_diversity(table)
    assert out['observed_taxa'] == 2
    assert out['shannon'] == pytest.approx(round(float(np.log(2)), 4), abs=1e-4)
    assert mg.compute_alpha_diversity(pd.DataFrame()) == {}


def test_compute_beta_diversity():
    import pandas as pd
    tables = [
        pd.DataFrame({'taxon': ['speciesA', 'speciesB'], 'count': [50, 50]}),
        pd.DataFrame({'taxon': ['speciesA'], 'count': [100]}),
    ]
    out = mg.compute_beta_diversity(tables)
    assert out.shape == (2, 2)
    assert out[0][1] > 0     # samples differ


# ── full classify path (pure-Python builtin) ─────────────────────────────────

def test_classify_reads_builtin(tmp_path):
    fq = tmp_path / 'reads.fq'
    seq = 'ATGCGTACGTAGCTAGCTAGATCGATCGATCG'
    fq.write_text(f"@r1\n{seq}\n+\n{'F' * len(seq)}\n")
    res = mg.classify_reads(str(fq))
    assert res is not None
    rep = mg.format_metagenomics_report(res)
    assert isinstance(rep, str) and len(rep) > 5


# ── 16S classification + identity helpers ────────────────────────────────────

def test_classify_16s_two_sequences():
    result = mg.classify_16s_rna([('s1', 'TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACC'),
                                  ('s2', 'AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAA')])
    assert result is not None
    txt = mg.format_16s_report(result)
    assert isinstance(txt, str) and len(txt) > 5


def test_compute_identity():
    s1 = 'ACGTACGTACGT' * 4
    s2 = ('ACGA' * 3 + 'ACGT' * 3) * 2
    ident = mg._compute_identity(s1, s2)
    assert 0.0 <= ident <= 1.0


def test_alpha_diversity_single():
    assert mg.alpha_diversity_single(['A', 'B', 'A', 'C'], method='shannon') > 0


def test_analyze_diversity_wrapper():
    import pandas as pd
    tables = [
        pd.DataFrame({'taxon': ['X', 'Y'], 'count': [5, 5]}),
        pd.DataFrame({'taxon': ['X', 'Y'], 'count': [9, 1]}),
    ]
    out = mg.analyze_diversity(tables)
    assert len(out.alpha_diversity) == 2
    assert out.beta_diversity is not None
