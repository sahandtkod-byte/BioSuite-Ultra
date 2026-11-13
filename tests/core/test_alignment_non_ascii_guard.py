"""The single-byte guard in front of the vectorised alignment fill.

`_match_array` reads both sequences with ``np.frombuffer(..., dtype='S1')``, so it
splits them into *bytes* while the DP grid is sized by ``len()`` — *characters*.
A sequence line carrying one multi-byte character therefore made the two shapes
disagree and numpy answered "operands could not be broadcast together with shapes
(4,) (5,)" from inside the fill loop.
"""
import pytest

from biosuite.core.alignment import (
    needleman_wunsch,
    smith_waterman,
    multiple_alignment,
)


@pytest.mark.parametrize("aligner", [needleman_wunsch, smith_waterman])
def test_a_multi_byte_character_in_the_second_sequence_is_named(aligner):
    with pytest.raises(ValueError) as exc:
        aligner("ACGT", "ACGξ")
    assert str(exc.value) == (
        "seq2 must hold single-byte residue codes, but character 'ξ' at position 3 "
        "is not one. Remove it or replace it with '-' for an unknown position.")


@pytest.mark.parametrize("aligner", [needleman_wunsch, smith_waterman])
def test_a_multi_byte_character_in_the_first_sequence_is_named(aligner):
    with pytest.raises(ValueError,
                       match=r"^seq1 must hold single-byte residue codes, but character "
                             r"'ξ' at position 2 is not one\."):
        aligner("ACξT", "ACGT")


@pytest.mark.parametrize("aligner", [needleman_wunsch, smith_waterman])
def test_a_rtl_character_reports_its_own_position(aligner):
    with pytest.raises(ValueError, match="character 'ن' at position 0"):
        aligner("ACGT", "نACG")


@pytest.mark.parametrize("gap, expected", [
    ("ACG-", 2),      # '-' is an ordinary mismatching character, not a wildcard
    ("ACGT", 4),
    ("", -8),         # every column is a gap in seq2
])
def test_single_byte_sequences_still_align(gap, expected):
    assert needleman_wunsch("ACGT", gap) == ("ACGT", "-" * (4 - len(gap)) + gap, expected)


def test_a_blank_pair_and_a_local_search_on_an_empty_sequence():
    assert needleman_wunsch("", "") == ("", "", 0)
    assert smith_waterman("", "AC") == ("", "", 0)


def test_multiple_alignment_refuses_a_multi_byte_member():
    """The progressive aligner reaches the guard through its pairwise step, so the
    bad sequence is still named rather than crashing the guide tree."""
    with pytest.raises(ValueError, match="seq2 must hold single-byte"):
        multiple_alignment(["ACGT", "ACξT", "ACG-T"])
