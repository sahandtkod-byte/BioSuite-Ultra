"""Regression tests for metagenomics.py review fixes."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.metagenomics import (
    _builtin_classify, compute_beta_diversity, classify_16s_rna,
    shannon_entropy, simpson_index, chao1_estimator, bray_curtis_distance,
    alpha_diversity_single, KNOWN_TAXA,
)


ECOLI = KNOWN_TAXA and [k for k, v in KNOWN_TAXA.items() if v == 'Escherichia coli'][0]
STAPH = [k for k, v in KNOWN_TAXA.items() if v == 'Staphylococcus aureus'][0]


def _fq(tmp_path, name, reads):
    p = tmp_path / name
    p.write_text(''.join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return str(p)


def test_per_read_voting_not_global_collapse(tmp_path):
    # 3 E. coli reads then 1 Staph read: the Staph read must stay Staph —
    # global-vote counting assigned it to E. coli (historical majority).
    ecoli_read = ECOLI + "A" * 20
    staph_read = "C" * 20 + STAPH
    path = _fq(tmp_path, "r.fq", [ecoli_read] * 3 + [staph_read])
    res = _builtin_classify(path)
    taxa = [c['taxon'] for c in res.classifications]
    assert taxa.count('Staphylococcus aureus') == 1
    assert taxa.count('Escherichia coli') == 3


def test_abundance_is_read_based_and_bounded(tmp_path):
    ecoli_read = ECOLI + "A" * 20
    path = _fq(tmp_path, "r.fq", [ecoli_read, ecoli_read])
    res = _builtin_classify(path)
    df = res.abundance_table
    assert df['relative_abundance'].sum() <= 100.0
    assert df.iloc[0]['count'] == 2
    assert df.iloc[0]['relative_abundance'] == 100.0


def test_beta_diversity_taxon_aligned():
    t1 = pd.DataFrame({'taxon': ['a', 'b'], 'count': [10, 0]})
    t2 = pd.DataFrame({'taxon': ['b', 'a'], 'count': [10, 0]})  # b-dominant, permuted order
    d = compute_beta_diversity([t1, t2])
    assert d[0, 1] == pytest.approx(1.0)   # fully disjoint, position-zip said 0
    t3 = pd.DataFrame({'taxon': ['a', 'b'], 'count': [10, 0]})
    d2 = compute_beta_diversity([t1, t3])
    assert d2[0, 1] == pytest.approx(0.0)


def test_diversity_formulas_known_answers():
    assert shannon_entropy([50, 50]) == pytest.approx(np.log(2))
    assert simpson_index([50, 50]) == pytest.approx(0.505, rel=1e-2)
    assert chao1_estimator([5, 4, 1]) == pytest.approx(3)  # f1=1, f2=0 -> 3+0
    assert chao1_estimator([5, 4, 2]) == pytest.approx(3)  # f2=1, f1=0 -> 3
    assert bray_curtis_distance([10, 0], [0, 10]) == pytest.approx(1.0)
    assert bray_curtis_distance([5, 5], [5, 5]) == pytest.approx(0.0)


def test_alpha_single_methods():
    reads = ['a', 'a', 'b', 'c']
    assert alpha_diversity_single(reads, 'observed') == 3
    assert alpha_diversity_single(reads, 'shannon') > 0
    assert alpha_diversity_single([], 'shannon') == 0.0


def test_16s_classifies_its_reference():
    from biosuite.core.metagenomics import SILVA_16S_DB
    seqs = [('q1', SILVA_16S_DB['Escherichia coli'])]
    res = classify_16s_rna(seqs)
    assert res.classifications[0]['taxonomy'] == 'Escherichia coli'
    assert res.classifications[0]['status'] == 'classified'
