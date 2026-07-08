"""
Unit tests for biosuite.core.msa module.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.msa import (
    auto_align, consensus_sequence, alignment_statistics,
    format_alignment, read_fasta_for_msa, _progressive_msa,
    _kmer_distance, _pairwise_distance
)


class TestProgressiveMSA:
    def test_two_sequences(self):
        seqs = [('A', 'ACGT'), ('B', 'ACGT')]
        result = _progressive_msa(seqs)
        assert len(result) == 2
        assert len(result[0][1]) == len(result[1][1])

    def test_three_sequences(self):
        seqs = [('A', 'ACGT'), ('B', 'ACGA'), ('C', 'ACGT')]
        result = _progressive_msa(seqs)
        assert len(result) == 3
        lengths = set(len(s) for _, s in result)
        assert len(lengths) == 1  # all same length

    def test_identical_sequences(self):
        seqs = [('A', 'ACGTACGT'), ('B', 'ACGTACGT')]
        result = _progressive_msa(seqs)
        assert result[0][1] == result[1][1]


class TestAutoAlign:
    def test_basic_alignment(self):
        seqs = [('seq1', 'ACGT'), ('seq2', 'ACGA')]
        result = auto_align(seqs)
        assert result.num_sequences == 2
        assert result.alignment_length > 0

    def test_single_sequence(self):
        result = auto_align([('seq1', 'ACGT')])
        assert 'at least 2' in result.message.lower()

    def test_empty_input(self):
        result = auto_align([])
        assert 'at least 2' in result.message.lower()


class TestConsensusSequence:
    def test_identical(self):
        from biosuite.core.msa import MSA
        msa = MSA(method='test', sequences=[('A', 'ACGT'), ('B', 'ACGT')],
                  num_sequences=2, alignment_length=4)
        consensus = consensus_sequence(msa)
        assert consensus == 'ACGT'

    def test_mixed(self):
        from biosuite.core.msa import MSA
        msa = MSA(method='test', sequences=[('A', 'ACGT'), ('B', 'ACGA')],
                  num_sequences=2, alignment_length=4)
        consensus = consensus_sequence(msa)
        assert len(consensus) == 4


class TestAlignmentStats:
    def test_basic_stats(self):
        from biosuite.core.msa import MSA
        msa = MSA(method='test', sequences=[('A', 'ACGT'), ('B', 'ACGT')],
                  num_sequences=2, alignment_length=4)
        stats = alignment_statistics(msa)
        assert stats['num_sequences'] == 2
        assert stats['alignment_length'] == 4


class TestReadFastaForMSA:
    def test_read(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">s1\nACGT\n>s2\nTTTT\n")
            f.flush()
            seqs = read_fasta_for_msa(f.name)
        os.unlink(f.name)
        assert len(seqs) == 2
        assert seqs[0] == ('s1', 'ACGT')
