"""Tests for biosuite.core.sequence module."""
import pytest


class TestGcContent:
    """Tests for gc_content()."""

    @pytest.mark.parametrize("seq,expected", [
        ("ATCG", 50.0),
        ("AAAA", 0.0),
        ("CCCC", 100.0),
        ("ATCGATCG", 50.0),
        ("GGGGCCCC", 100.0),
        ("", 0.0),
        ("ATCGATCGATCGATCG", 50.0),
    ])
    def test_gc_content_values(self, seq, expected):
        from biosuite.core.sequence import gc_content
        assert gc_content(seq) == expected

    def test_gc_content_case_insensitive(self):
        from biosuite.core.sequence import gc_content
        assert gc_content("atcg") == gc_content("ATCG")

    def test_gc_content_empty(self):
        from biosuite.core.sequence import gc_content
        assert gc_content("") == 0.0


class TestReverseComplement:
    """Tests for reverse_complement()."""

    @pytest.mark.parametrize("seq,expected", [
        ("ATCG", "CGAT"),
        ("AAAA", "TTTT"),
        ("ACGT", "ACGT"),  # palindrome
        ("", ""),
        ("aTcG", "CgAt"),  # preserves case
    ])
    def test_reverse_complement(self, seq, expected):
        from biosuite.core.sequence import reverse_complement
        assert reverse_complement(seq) == expected

    def test_reverse_complement_is_involutive(self):
        """Reverse complement applied twice returns original."""
        from biosuite.core.sequence import reverse_complement
        seq = "ATCGATCG"
        assert reverse_complement(reverse_complement(seq)) == seq


class TestTranslate:
    """Tests for translate()."""

    @pytest.mark.parametrize("dna,expected", [
        ("ATGAAATTTTAA", "MKF*"),
        ("ATG", "M"),
        ("ATGTAATAA", "M**"),   # stop codons
        ("", ""),
    ])
    def test_translate(self, dna, expected):
        from biosuite.core.sequence import translate
        assert translate(dna) == expected

    def test_translate_frame_2(self):
        from biosuite.core.sequence import translate
        # "A ATG GAA TTT TAA" in frame 2 -> "M F *" 
        result = translate("XATGAAATTTTAA", frame=2)
        assert result.startswith("M")


class TestSequenceStats:
    """Tests for sequence_stats()."""

    def test_basic_stats(self):
        from biosuite.core.sequence import sequence_stats
        result = sequence_stats("ATCG")
        assert result["length"] == 4
        assert result["A"] == 1
        assert result["T"] == 1
        assert result["C"] == 1
        assert result["G"] == 1
        assert result["N"] == 0

    def test_empty_stats(self):
        from biosuite.core.sequence import sequence_stats
        result = sequence_stats("")
        assert result["length"] == 0

    def test_all_n(self):
        from biosuite.core.sequence import sequence_stats
        result = sequence_stats("NNNN")
        assert result["N"] == 4
        assert result["GC"] == 0.0


class TestFastaReading:
    """Tests for read_fasta()."""

    def test_read_fasta(self, fasta_file):
        from biosuite.core.sequence import read_fasta
        result = read_fasta(fasta_file)
        assert result is not None
        assert len(result) == 2
        assert result[0][0] == "seq1 test sequence 1"
        # Sequence is joined across lines
        assert "ATCGATCGATCGATCGATCGATCGATCGATCG" in result[0][1]

    def test_read_fasta_nonexistent(self):
        from biosuite.core.sequence import read_fasta
        result = read_fasta("/nonexistent/file.fasta")
        assert result is None


class TestFastqReading:
    """Tests for read_fastq()."""

    def test_read_fastq(self, fastq_file):
        from biosuite.core.sequence import read_fastq
        result = read_fastq(fastq_file)
        assert result is not None
        assert len(result) == 2
        assert result[0][0] == "read1"
        assert result[0][1] == "ATCGATCGATCG"
