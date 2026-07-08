"""
Unit tests for biosuite.core.phylogeny module.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.phylogeny import p_distance, distance_matrix, upgma_tree


# ─── p_distance ──────────────────────────────────────────────────────────────

class TestPDistance:
    def test_identical(self):
        assert p_distance("AAAA", "AAAA") == 0.0

    def test_all_different(self):
        assert p_distance("AAAA", "TTTT") == 1.0

    def test_half(self):
        assert p_distance("AATT", "AACC") == 0.5

    def test_with_gaps_excluded(self):
        # Gaps are excluded: position 1 has a gap, so only positions 0 and 2 count
        # A-A vs ATA: pos 0 (A,A) match, pos 1 (-,T) excluded, pos 2 (A,A) match
        assert p_distance("A-A", "ATA") == 0.0  # 0 diffs / 2 valid positions

    def test_all_gaps(self):
        assert p_distance("----", "ACGT") == 0.0  # no comparable positions

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            p_distance("AAAA", "AA")

    def test_single_position(self):
        assert p_distance("A", "A") == 0.0
        assert p_distance("A", "T") == 1.0


# ─── distance_matrix ─────────────────────────────────────────────────────────

class TestDistanceMatrix:
    def test_two_identical(self):
        mat = distance_matrix(["AAAA", "AAAA"])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0
        assert mat[1, 1] == 0.0
        assert mat[0, 1] == 0.0
        assert mat[1, 0] == 0.0

    def test_two_different(self):
        mat = distance_matrix(["AAAA", "TTTT"])
        assert mat[0, 1] == 1.0
        assert mat[1, 0] == 1.0

    def test_symmetric(self):
        seqs = ["ACGT", "TGCA", "AAAA"]
        mat = distance_matrix(seqs)
        assert np.allclose(mat, mat.T)

    def test_diagonal_zero(self):
        seqs = ["ACGT", "TGCA", "AAAA", "CCCC"]
        mat = distance_matrix(seqs)
        for i in range(4):
            assert mat[i, i] == 0.0

    def test_three_sequences(self):
        mat = distance_matrix(["AAAA", "AAAT", "AATT"])
        assert mat.shape == (3, 3)
        # AAAA vs AAAT: 1 diff / 4 total = 0.25
        assert mat[0, 1] == pytest.approx(0.25)
        # AAAA vs AATT: 2 diff / 4 total = 0.5
        assert mat[0, 2] == pytest.approx(0.5)


# ─── upgma_tree ──────────────────────────────────────────────────────────────

class TestUPGMATree:
    def test_returns_linkage(self):
        mat = np.array([[0, 0.25, 0.5],
                        [0.25, 0, 0.5],
                        [0.5, 0.5, 0]])
        labels = ["A", "B", "C"]
        link = upgma_tree(mat, labels)
        assert link is not None
        assert link.shape[1] == 4  # scipy linkage format

    def test_two_sequences(self):
        mat = np.array([[0, 0.3], [0.3, 0]])
        labels = ["A", "B"]
        link = upgma_tree(mat, labels)
        assert link.shape[0] == 1  # 1 merge for 2 sequences
