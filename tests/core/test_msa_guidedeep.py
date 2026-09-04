"""MSA internals: k-mer distances, guide tree, profile DP alignment, merging."""
import numpy as np
import pytest

from biosuite.core import msa as ms


S_seqs = ['ACGTACGT', 'ACGTACGA', 'ACGTTCGA', 'TCGTTCGA']


def test_pairwise_distance_symmetric():
    d = np.asarray(ms._pairwise_distance(S_seqs))
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert d.shape == (4, 4)


def test_kmer_distance_identical_vs_different():
    assert ms._kmer_distance('ACGTACGT', 'ACGTACGT') == 0 or \
        ms._kmer_distance('ACGTACGT', 'ACGTACGT') < 1e-9
    assert ms._kmer_distance('ACGTACGT', 'TTTTAAAA') > 0


def test_upgma_tree_returns_topology():
    d = ms._pairwise_distance(S_seqs)
    tree = ms._upgma_tree(d, len(S_seqs))
    assert tree is not None


def test_build_guide_order_returns_all_indices():
    d = ms._pairwise_distance(S_seqs)
    order = ms._build_guide_order(d, [f's{i}' for i in range(4)])
    assert isinstance(order, list)
    flat = set(order) if order and isinstance(order[0], int) else set(
        i for pair in order for i in pair)
    assert flat == {0, 1, 2, 3}


def test_align_two_profiles_perfect():
    a, b = ms._align_two_profiles('ACGT', 'ACGT')
    assert a == 'ACGT' and b == 'ACGT'


def test_align_two_profiles_with_gap_prefers_identity_when_possible():
    a, b = ms._align_two_profiles('AAAA', 'AA')
    assert len(a) == len(b) == 4
    assert b.count('-') == 2
    gn = a.replace('-', '')
    assert gn == 'AAAA'


def test_progressive_msa_equal_length():
    out = ms._progressive_msa(S_seqs)
    assert len({len(s) for s in out}) == 1


def test_consensus_and_statistics():
    aligned = ms.auto_align(S_seqs)
    cons = ms.consensus_sequence(aligned)
    stats = ms.alignment_statistics(aligned)
    seq0 = aligned.sequences[0][1] if isinstance(aligned.sequences[0], tuple) else aligned.sequences[0]
    assert len(cons) == len(seq0)
    assert isinstance(stats, dict) and stats


def test_is_nucleotide_tricky():
    assert ms._is_nucleotide('ACGTGGC') is True
    assert ms._is_nucleotide('MVLSPADKTNVKAAWG') is False


def test_auto_align_method_override():
    aligned = ms.auto_align(S_seqs, method='builtin')
    assert len(aligned.sequences) == 4
