"""
Unit tests for biosuite.core.blast module.
"""
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.blast import (
    _build_kmer_index, _find_seed_hits, _read_fasta,
    run_blast, format_blast_result, BlastResult, BlastHit
)


class TestKmerIndex:
    def test_basic_index(self):
        seqs = [('seq1', 'ACGTACGT')]
        index = _build_kmer_index(seqs, k=4)
        assert len(index) > 0
        assert 'ACGT' in index or 'CGTA' in index or 'GTAC' in index

    def test_empty_sequence(self):
        index = _build_kmer_index([('seq1', '')], k=4)
        assert len(index) == 0

    def test_with_n_bases(self):
        seqs = [('seq1', 'ACGTNNACGT')]
        index = _build_kmer_index(seqs, k=4)
        # N-containing kmers should be excluded
        for kmer in index:
            assert 'N' not in kmer


class TestSeedHits:
    def test_exact_match(self):
        seqs = [('db', 'ACGTACGTACGT')]
        index = _build_kmer_index(seqs, k=4)
        hits = _find_seed_hits('ACGT', index, k=4)
        assert len(hits) > 0

    def test_no_match(self):
        seqs = [('db', 'TTTTTTTT')]
        index = _build_kmer_index(seqs, k=4)
        hits = _find_seed_hits('ACGT', index, k=4)
        assert len(hits) == 0


class TestReadFasta:
    def test_single_record(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">seq1\nACGTACGT\n")
            f.flush()
            result = _read_fasta(f.name)
        os.unlink(f.name)
        assert len(result) == 1
        assert result[0][0] == 'seq1'
        assert result[0][1] == 'ACGTACGT'

    def test_multi_record(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">s1\nACGT\n>s2\nTTTT\n")
            f.flush()
            result = _read_fasta(f.name)
        os.unlink(f.name)
        assert len(result) == 2


class TestBlastSearch:
    def test_basic_search(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as db:
            db.write(">ref\nACGTACGTACGTACGTACGT\n")
            db.flush()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as q:
                q.write(">query\nACGTACGTACGT\n")
                q.flush()
                result = run_blast(q.name, db.name, evalue=1)
        os.unlink(db.name)
        os.unlink(q.name)
        assert isinstance(result, BlastResult)
        assert result.engine == 'builtin'

    def test_no_match(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as db:
            db.write(">ref\nTTTTTTTTTTTT\n")
            db.flush()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as q:
                q.write(">query\nACGTACGT\n")
                q.flush()
                result = run_blast(q.name, db.name, evalue=0.01)
        os.unlink(db.name)
        os.unlink(q.name)
        assert result.num_hits == 0

    def test_format_result(self):
        result = BlastResult(
            engine='builtin', program='test', database='test.db',
            query_length=100, hits=[]
        )
        formatted = format_blast_result(result)
        assert 'builtin' in formatted
        assert '0' in formatted
