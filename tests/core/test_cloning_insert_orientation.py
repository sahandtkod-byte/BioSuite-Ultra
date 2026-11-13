"""What `verify_insert` is allowed to conclude about orientation.

Two claims used to be wrong in opposite directions.  A self-complementary insert
is one string read either way, yet it was reported as "found in BOTH orientations"
and told the user to resolve it by restriction mapping.  And a truncated insert
was only searched in the forward reading, so the commonest failure -- a partial
insert in the reverse orientation -- came back as a bare "not found".
"""
import pytest

from biosuite.core import cloning as cl

PALINDROME = "GAATTCGAATTC"                     # 12 nt, reads the same both ways
# A 16 nt self-complementary insert that is not itself a repetition, so a search
# for it finds exactly the copies that were ligated.
X = "ATGCCGTA"
CLEAN_PALINDROME = X + cl._reverse_complement(X)
INSERT = "ACGTACGTACGTTTGCAATCGATCGATCGATC"      # 32 nt, not palindromic


def test_the_two_palindromic_inserts_are_their_own_reverse_complement():
    assert cl._reverse_complement(PALINDROME) == PALINDROME
    assert cl._reverse_complement(CLEAN_PALINDROME) == CLEAN_PALINDROME
    assert cl._reverse_complement(INSERT) != INSERT


def test_a_self_complementary_insert_reports_that_orientation_cannot_be_read():
    res = cl.verify_insert("TTTT" + PALINDROME + "CCCC", PALINDROME)
    assert res["insert_found"] is True
    assert res["forward_orientation"] is True
    assert (res["insert_start"], res["insert_end"]) == (4, 16)
    assert res["warnings"] == [
        "The insert is self-complementary, so its orientation in the product "
        "cannot be distinguished."]


def test_a_reverse_oriented_insert_is_reported_as_reverse_without_comment():
    rc = cl._reverse_complement(INSERT)
    res = cl.verify_insert("TTTT" + rc + "CCCC", INSERT)
    assert res["insert_found"] is True
    assert res["forward_orientation"] is False
    assert (res["insert_start"], res["insert_end"]) == (4, 36)
    assert res["warnings"] == []


def test_a_half_insert_in_the_reverse_reading_is_named_as_such():
    half = len(INSERT) // 2
    rc = cl._reverse_complement(INSERT)
    res = cl.verify_insert("TTTT" + rc[:half] + "GGGG", INSERT)
    assert res["insert_found"] is False
    assert (res["insert_start"], res["insert_end"]) == (None, None)
    assert res["warnings"] == [
        "Insert NOT found in the product sequence.",
        "Only the 5' half of the insert read backwards (first 16 bp) was found. "
        "Insert may be truncated and reverse-oriented."]


def test_the_other_half_of_a_reverse_oriented_truncation_is_distinguished():
    half = len(INSERT) // 2
    rc = cl._reverse_complement(INSERT)
    res = cl.verify_insert("TTTT" + rc[half:] + "GGGG", INSERT)
    assert res["warnings"][-1] == (
        "Only the 3' half of the insert read backwards (last 16 bp) was found. "
        "Insert may be truncated and reverse-oriented.")


def test_a_forward_oriented_truncation_keeps_its_original_wording():
    res = cl.verify_insert("TTTT" + INSERT[:16] + "GGGG", INSERT)
    assert res["warnings"] == [
        "Insert NOT found in the product sequence.",
        "Only the 5' half of the insert (first 16 bp) was found. "
        "Insert may be truncated."]


def test_a_short_insert_is_never_searched_in_halves():
    # The half-splitting heuristic needs > 20 nt, so a 12 nt loss is only "absent".
    res = cl.verify_insert("TTTT" + "CCCC", INSERT[:12])
    assert res["warnings"] == ["Insert NOT found in the product sequence."]
