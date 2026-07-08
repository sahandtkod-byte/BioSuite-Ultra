"""
Unit tests for biosuite.core.alignment module.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.alignment import needleman_wunsch, smith_waterman, _match_array


# ─── Needleman-Wunsch ────────────────────────────────────────────────────────

class TestNeedlemanWunsch:
    def test_identical_sequences(self):
        a1, a2, score = needleman_wunsch("ACGT", "ACGT")
        assert a1 == "ACGT"
        assert a2 == "ACGT"
        assert score == 4  # 4 matches * 1

    def test_single_mismatch(self):
        a1, a2, score = needleman_wunsch("ACGT", "ACAT")
        assert "A" in a1
        assert "C" in a1
        assert score < 4

    def test_with_gap(self):
        a1, a2, score = needleman_wunsch("ACGT", "ACG")
        assert len(a1) == 4
        assert len(a2) == 4
        assert "-" in a2

    def test_empty_sequence(self):
        a1, a2, score = needleman_wunsch("", "ACGT")
        assert a1 == "----"
        assert a2 == "ACGT"

    def test_both_empty(self):
        a1, a2, score = needleman_wunsch("", "")
        assert a1 == ""
        assert a2 == ""

    def test_single_char(self):
        a1, a2, score = needleman_wunsch("A", "A")
        assert score == 1

    def test_symmetric_length(self):
        a1, a2, score = needleman_wunsch("ACGT", "TGCA")
        assert len(a1) == len(a2)

    def test_longer_sequences(self):
        seq1 = "ACGTACGTACGT"
        seq2 = "ACGTACGTACGT"
        a1, a2, score = needleman_wunsch(seq1, seq2)
        assert a1 == seq1
        assert a2 == seq2
        assert score == 12

    def test_score_with_gaps(self):
        a1, a2, score = needleman_wunsch("AAAA", "TTTT", match=1, mismatch=-1, gap=-2)
        # All mismatches gives -4, but gaps might produce a higher score
        assert score <= 0  # All negative or zero for completely different sequences


# ─── Smith-Waterman ──────────────────────────────────────────────────────────

class TestSmithWaterman:
    def test_identical_sequences(self):
        a1, a2, score = smith_waterman("ACGT", "ACGT")
        assert a1 == "ACGT"
        assert a2 == "ACGT"
        assert score == 4

    def test_local_alignment(self):
        # "CGT" is shared, flanked by different bases
        a1, a2, score = smith_waterman("XXCGTXX", "YYCGTYY")
        assert score >= 3  # At least 3 matches
        assert "CGT" in a1 or "CGT" in a2

    def test_no_alignment(self):
        # Completely different, short sequences
        a1, a2, score = smith_waterman("A", "T")
        assert score == 0  # No positive match

    def test_empty(self):
        a1, a2, score = smith_waterman("", "ACGT")
        assert score == 0

    def test_score_non_negative(self):
        _, _, score = smith_waterman("ACGT", "TGCA")
        assert score >= 0

    def test_local_substring(self):
        a1, a2, score = smith_waterman("TTACGTTT", "XXACGXX")
        assert score >= 3  # "ACG" = 3 matches

    def test_longer_sequences(self):
        seq1 = "AAAAAAAAACGT"
        seq2 = "TTTTTTTTACGT"
        a1, a2, score = smith_waterman(seq1, seq2)
        assert score >= 4  # "ACGT" at end


# ─── _match_array ────────────────────────────────────────────────────────────

class TestMatchArray:
    def test_identical(self):
        arr = _match_array("AA", "AA", match=1, mismatch=-1)
        assert arr.shape == (2, 2)
        assert np.all(arr == 1)

    def test_all_mismatch(self):
        arr = _match_array("AT", "CG", match=1, mismatch=-1)
        assert np.all(arr == -1)

    def test_mixed(self):
        arr = _match_array("AC", "AG", match=2, mismatch=-3)
        assert arr[0, 0] == 2   # A==A (match)
        assert arr[0, 1] == -3  # A!=G (mismatch)
        assert arr[1, 0] == -3  # C!=A (mismatch)
        assert arr[1, 1] == -3  # C!=G (mismatch)

    def test_shape(self):
        arr = _match_array("ACGT", "ACGT", 1, -1)
        assert arr.shape == (4, 4)
