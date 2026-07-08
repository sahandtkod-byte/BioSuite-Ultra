"""
Unit tests for biosuite.core.sequence module.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.sequence import (
    read_fasta, read_fastq, read_genbank,
    gc_content, reverse_complement, translate,
    sequence_stats, quality_stats
)


# ─── gc_content ──────────────────────────────────────────────────────────────

class TestGCContent:
    def test_all_gc(self):
        assert gc_content("GCGC") == 100.0

    def test_all_at(self):
        assert gc_content("ATAT") == 0.0

    def test_mixed(self):
        assert gc_content("ACGT") == 50.0

    def test_empty(self):
        assert gc_content("") == 0.0

    def test_lowercase(self):
        assert gc_content("gcgc") == 100.0

    def test_with_n(self):
        assert gc_content("GCGCNN") == pytest.approx(66.67, abs=0.01)

    def test_single_base(self):
        assert gc_content("G") == 100.0
        assert gc_content("A") == 0.0


# ─── reverse_complement ──────────────────────────────────────────────────────

class TestReverseComplement:
    def test_simple(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self):
        assert reverse_complement("AATT") == "AATT"

    def test_lowercase(self):
        assert reverse_complement("atcg") == "cgat"

    def test_with_n(self):
        assert reverse_complement("ATNG") == "CNAT"

    def test_empty(self):
        assert reverse_complement("") == ""

    def test_single(self):
        assert reverse_complement("A") == "T"
        assert reverse_complement("C") == "G"

    def test_roundtrip(self):
        seq = "ACGTACGTNNNN"
        assert reverse_complement(reverse_complement(seq)) == seq.upper()


# ─── translate ───────────────────────────────────────────────────────────────

class TestTranslate:
    def test_atg_start(self):
        assert translate("ATG") == "M"

    def test_stop_codon(self):
        assert translate("TAA") == "*"
        assert translate("TAG") == "*"
        assert translate("TGA") == "*"

    def test_known_codons(self):
        # TTT -> F, TCT -> S, TAT -> Y, TGT -> C
        assert translate("TTT") == "F"
        assert translate("TCT") == "S"
        assert translate("TAT") == "Y"
        assert translate("TGT") == "C"

    def test_longer_sequence(self):
        # ATG GCT TAA -> M A *
        assert translate("ATGGCTTAA") == "MA*"

    def test_frame_1(self):
        result = translate("ATCGATCG", frame=1)
        assert result[0] == "I"  # ATC -> I
        assert len(result) == 2  # 8 bases = 2 codons

    def test_invalid_codon_returns_x(self):
        assert translate("NNN") == "X"

    def test_empty(self):
        assert translate("") == ""


# ─── sequence_stats ──────────────────────────────────────────────────────────

class TestSequenceStats:
    def test_basic(self):
        stats = sequence_stats("ATCG")
        assert stats['length'] == 4
        assert stats['A'] == 1
        assert stats['T'] == 1
        assert stats['G'] == 1
        assert stats['C'] == 1
        assert stats['N'] == 0

    def test_gc_percentage(self):
        stats = sequence_stats("GCGC")
        assert stats['GC'] == 100.0

    def test_at_percentage(self):
        stats = sequence_stats("ATAT")
        assert stats['AT'] == 100.0

    def test_empty_sequence(self):
        stats = sequence_stats("")
        assert stats['length'] == 0
        assert stats['A'] == 0
        assert stats['GC'] == 0.0

    def test_with_n(self):
        stats = sequence_stats("ATCGNNNN")
        assert stats['N'] == 4
        assert stats['length'] == 8

    def test_lowercase_handled(self):
        stats = sequence_stats("atcg")
        assert stats['A'] == 1
        assert stats['T'] == 1


# ─── quality_stats ───────────────────────────────────────────────────────────

class TestQualityStats:
    def test_high_quality(self):
        # Phred+33: 'I' = ord('I') - 33 = 73 - 33 = 40
        qs = quality_stats("IIII")
        assert qs['mean'] == 40.0
        assert qs['min'] == 40
        assert qs['max'] == 40

    def test_low_quality(self):
        # '!' = ord('!') - 33 = 33 - 33 = 0
        qs = quality_stats("!!!!")
        assert qs['mean'] == 0.0

    def test_positions(self):
        qs = quality_stats("IIII")
        assert qs['positions'] == [1, 2, 3, 4]

    def test_mixed_quality(self):
        # '!' = 0, 'I' = 40
        qs = quality_stats("!I")
        assert qs['mean'] == 20.0
        assert qs['min'] == 0
        assert qs['max'] == 40


# ─── read_fasta ──────────────────────────────────────────────────────────────

class TestReadFasta:
    def test_single_sequence(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">seq1\nACGTACGT\n")
        result = read_fasta(str(f))
        assert len(result) == 1
        assert result[0][0] == "seq1"
        assert result[0][1] == "ACGTACGT"

    def test_multi_sequence(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">s1\nACGT\n>s2\nTTTT\n")
        result = read_fasta(str(f))
        assert len(result) == 2

    def test_multiline_sequence(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">seq1\nACGT\nTGCA\n")
        result = read_fasta(str(f))
        assert result[0][1] == "ACGTTGCA"

    def test_nonexistent_file(self):
        assert read_fasta("nonexistent.fasta") is None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.fasta"
        f.write_text("")
        result = read_fasta(str(f))
        assert result == []


# ─── read_fastq ──────────────────────────────────────────────────────────────

class TestReadFastq:
    def test_single_record(self, tmp_path):
        f = tmp_path / "test.fastq"
        f.write_text("@read1\nACGT\n+\nIIII\n")
        result = read_fastq(str(f))
        assert len(result) == 1
        assert result[0][0] == "read1"
        assert result[0][1] == "ACGT"
        assert result[0][2] == "IIII"

    def test_nonexistent_file(self):
        assert read_fastq("nonexistent.fastq") is None
