"""Comprehensive MSA tests: engines, distance/tree helpers, stats, formats."""
import pytest

from biosuite.core import msa


SEQS = ['ACGTAAGG', 'ACGTAAG', 'ACGTAAGC']


def test_is_nucleotide():
    assert msa._is_nucleotide('ACGTN')
    assert not msa._is_nucleotide('MKHDLE')


def test_kmer_distance_zero_for_identical():
    assert msa._kmer_distance('ACGTACGT', 'ACGTACGT', k=2) == pytest.approx(0.0)
    d = msa._kmer_distance('AAAA', 'TTTT', k=2)
    assert d > 0


def test_pairwise_distance_symmetric():
    D = msa._pairwise_distance(SEQS, k=2)
    n = len(SEQS)
    assert len(D) == n
    for i in range(n):
        for j in range(n):
            assert D[i][j] == pytest.approx(D[j][i])
            if i == j:
                assert D[i][i] == pytest.approx(0.0)


def test_upgma_tree_groups_similar():
    D = [[0, 0.1, 0.9], [0.1, 0, 0.9], [0.9, 0.9, 0]]
    tree = msa._upgma_tree(D, 3)
    assert tree is not None


def test_build_guide_order_covers_all():
    D = [[0, 0.1, 0.9], [0.1, 0, 0.85], [0.9, 0.85, 0]]
    order = msa._build_guide_order(D, ['a', 'b', 'c'])
    assert sorted(order) == [0, 1, 2]


def test_align_two_profiles_basic():
    a, b = msa._align_two_profiles('ACGT', 'AGT')
    assert '-' in b or '-' in a
    assert len(a) == len(b)
    assert a.replace('-', '') == 'ACGT'
    assert b.replace('-', '') == 'AGT'


def test_merge_alignments_equal_lengths():
    out = msa._merge_alignments(['AC-T', 'AC-T'], ['AGT'])
    assert all(len(s) == len(out[0]) for s in out)


def test_progressive_msa_returns_equal_lengths():
    result = msa._progressive_msa([('s1', 'MKLL'), ('s2', 'MKLVV'), ('s3', 'MKTAAT')])
    lens = {len(s) for _, s in result}
    assert len(lens) == 1


def test_auto_align_builtin():
    m = msa.auto_align(SEQS, method='builtin')
    assert isinstance(m, msa.MSA)
    assert [n for n, _ in m.sequences] == ['seq1', 'seq2', 'seq3']
    lens = {len(s) for _, s in m.sequences}
    assert len(lens) == 1


def test_auto_align_handles_single_sequence():
    # degenerate case is handled gracefully (no crash)
    r = msa.auto_align(['ONLYONE'], method='builtin')
    assert isinstance(r, msa.MSA)


def test_consensus_threshold():
    # build via auto_align so internal counters (num_sequences etc.) are valid
    m = msa.auto_align(['ACGT', 'ACGT', 'ACTT'], method='builtin')
    assert msa.consensus_sequence(m, threshold=0.5).upper() == 'ACGT'


def test_alignment_statistics_fields():
    m = msa.auto_align(SEQS, method='builtin')
    stats = msa.alignment_statistics(m)
    assert stats['num_sequences'] == 3
    assert 0 <= stats['gap_percentage'] <= 100


def test_compute_conservation_range():
    m = msa.auto_align(SEQS, method='builtin')
    cons = msa.compute_conservation(m)
    assert all(0.0 <= c <= 1.0 for c in cons)


def test_format_alignment_text():
    m = msa.auto_align(SEQS, method='builtin')
    txt = msa.format_alignment(m, max_width=20, show_conservation=True)
    assert isinstance(txt, str) and 'seq1' in txt


def test_read_fasta_for_msa(tmp_path):
    p = tmp_path / 'in.fa'
    p.write_text(">a\nACGT\n>b\nAGGT\n")
    seqs = msa.read_fasta_for_msa(str(p))
    assert len(seqs) == 2


def test_write_fasta_roundtrip(tmp_path):
    p = tmp_path / 'out.fa'
    msa._write_fasta([('x', 'ACGT'), ('y', 'AGGT')], str(p))
    back = msa.read_fasta_for_msa(str(p))
    assert dict(back) == {'x': 'ACGT', 'y': 'AGGT'}
