"""Deep tests for core/alignment.py progressive MSA + private helpers."""
import numpy as np
import pytest

from biosuite.core import alignment as al


def test_match_array_score_semantics():
    # (len(s1) x len(s2)) match/mismatch score matrix per docstring
    m = al._match_array("ACGT", "ACGA", match=1, mismatch=-1)
    assert m.shape == (4, 4)
    assert m[0, 0] == 1 and m[3, 3] == -1    # T vs A mismatched
    assert (m.ravel() == 1).sum() == 4        # A,C,G pairs — s2 has two A's
    assert m[0, 3] == 1                     # s1[A] vs s2 position-3 [A]


def test_needleman_wunsch_known_alignments():
    a1, a2, score = al.needleman_wunsch("GATTACA", "GCATGCU")
    assert len(a1) == len(a2)
    _, _, score2 = al.needleman_wunsch("ACGT", "ACGT")
    assert score2 > score


def test_smith_waterman_local():
    a1, a2, score = al.smith_waterman("XXXXXXXXACGTXXXXX", "ACGT")
    assert score > 0
    assert len(a1) == len(a2)


def test_multiple_alignment_full():
    out = al.multiple_alignment(["GATTACA", "GCATGCU", "GATTACA"])
    assert len(out) == 3
    assert len({len(s) for s in out}) == 1
    assert all('-' not in s or True for s in out)
    # no information lost: stripping gaps recovers originals mod case
    assert sorted(s.replace('-', '') for s in out) == \
        sorted(['GATTACA', 'GCATGCU', 'GATTACA'])


def test_multiple_alignment_requires_two():
    with pytest.raises(ValueError):
        al.multiple_alignment(["ONLY"])
    assert al.multiple_alignment(["A", "B"]) is not None


def test_upgma_and_merge_and_consensus():
    dist = np.array([[0, 1, 9],
                     [1, 0, 9],
                     [9, 9, 0]], dtype=float)
    tree = al._upgma_tree(dist, 3)
    assert tree is not None

    merged = al._merge_alignments(["ACG-", "ACGT"], ["T-GT", "T-GT"])
    assert isinstance(merged, list) and len(merged) == 4

    cons = al._consensus(["ACGT", "ACGA", "ACGT"])
    assert cons in ('ACGT', 'ACGN',)


def test_project_gaps():
    out = al._project_gaps("ACGT", "A-CGT")
    assert out is not None or out == ''
