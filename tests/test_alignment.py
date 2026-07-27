"""Tests for biosuite.core.alignment module."""
import pytest


class TestNeedlemanWunsch:
    """Tests for needleman_wunsch()."""

    @pytest.mark.parametrize("s1,s2", [
        ("ACGT", "ACGT"),       # identical
        ("ACGT", "AACGT"),      # insertion
        ("ACGT", "CGT"),        # deletion
        ("AAAA", "TTTT"),       # all mismatches
        ("ACGT", "ACG"),        # prefix
    ])
    def test_nw_returns_tuple(self, s1, s2):
        from biosuite.core.alignment import needleman_wunsch
        a1, a2, score = needleman_wunsch(s1, s2)
        assert len(a1) == len(a2)
        assert isinstance(score, (int, float))

    def test_nw_identical(self):
        from biosuite.core.alignment import needleman_wunsch
        a1, a2, score = needleman_wunsch("ACGT", "ACGT")
        assert a1 == a2 == "ACGT"

    def test_nw_symmetric_score(self):
        """Score should be the same regardless of argument order."""
        from biosuite.core.alignment import needleman_wunsch
        _, _, score1 = needleman_wunsch("ACGT", "AACGT")
        _, _, score2 = needleman_wunsch("AACGT", "ACGT")
        assert score1 == score2


class TestSmithWaterman:
    """Tests for smith_waterman()."""

    @pytest.mark.parametrize("s1,s2", [
        ("ACGT", "ACGT"),
        ("ACGTACGT", "CGT"),
        ("AAAA", "TTTT"),
    ])
    def test_sw_returns_tuple(self, s1, s2):
        from biosuite.core.alignment import smith_waterman
        a1, a2, score = smith_waterman(s1, s2)
        assert len(a1) == len(a2)
        assert isinstance(score, (int, float))

    def test_sw_local_alignment(self):
        """SW should find the best local match."""
        from biosuite.core.alignment import smith_waterman
        a1, a2, score = smith_waterman("TTTTACGTTTTT", "ACGT")
        assert score > 0
