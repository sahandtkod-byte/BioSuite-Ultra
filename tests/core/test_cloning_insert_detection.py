"""Where `verify_insert` reports a located insert.

The old end-to-end file asserted only that the result dict was truthy.  A
verification tool is worth nothing unless its coordinates are the coordinates of
the copy it found, so the numbers below are the offsets in the products shown.
"""
from biosuite.core import cloning as cl

INSERT = "GCTAGCTAAGGTCC"             # 14 nt, not self-complementary
PALINDROME = "ACGTACGT"               # 8 nt, reads the same backwards


def test_a_forward_insert_is_located_between_its_flanks():
    res = cl.verify_insert("ATG" + INSERT + "TAA", INSERT)
    assert res["insert_found"] is True
    assert res["forward_orientation"] is True
    assert (res["insert_start"], res["insert_end"]) == (3, 17)
    assert res["warnings"] == []


def test_the_reported_span_is_exactly_the_insert_that_was_handed_in():
    product = "ATG" + INSERT + "TAA"
    res = cl.verify_insert(product, INSERT)
    assert product[res["insert_start"]:res["insert_end"]] == INSERT


def test_a_missing_short_insert_reports_no_position_and_one_warning():
    # "GGGGCCCC" is its own reverse complement, so the absence path is taken
    # through the self-complementary branch and may not claim an orientation.
    res = cl.verify_insert("ATGAAATAA", "GGGGCCCC")
    assert res["insert_found"] is False
    assert res["forward_orientation"] is False
    assert (res["insert_start"], res["insert_end"]) == (None, None)
    assert res["warnings"] == ["Insert NOT found in the product sequence."]


def test_a_missing_non_palindromic_insert_is_reported_the_same_way():
    res = cl.verify_insert("ATGAAATAA", "GCTAGCTA")
    assert res["insert_found"] is False
    assert res["forward_orientation"] is False
    assert res["warnings"] == ["Insert NOT found in the product sequence."]


def test_a_palindromic_insert_is_located_at_its_first_occurrence():
    res = cl.verify_insert("ATG" + PALINDROME + "TAA", PALINDROME)
    assert (res["insert_start"], res["insert_end"]) == (3, 11)
    assert res["forward_orientation"] is True


def test_an_insert_that_is_the_whole_product_is_still_located():
    res = cl.verify_insert(INSERT, INSERT)
    assert (res["insert_start"], res["insert_end"]) == (0, len(INSERT))
    assert res["warnings"] == []
