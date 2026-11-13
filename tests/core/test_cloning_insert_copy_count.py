"""How many copies of the insert a product carries, and what that does to the span.

A head-to-tail duplication is an ordinary ligation outcome.  `verify_insert`
looked at the first copy only, so the second copy was read as leftover vector and
the report concluded that "the product carries more vector than was handed in" --
about a construct that contains no extra vector at all.
"""
import pytest

from biosuite.core import cloning as cl

INSERT = "ACGTACGTACGTTTGCAATCGATCGATCGATC"      # 32 nt
BACKBONE_INS = "ACGTTGCAGATTACGTACGTGATCCGGA"    # 28 nt
BACKBONE = "TTTT" * 12                           # 48 nt
X = "ATGCCGTA"
CLEAN_PALINDROME = X + cl._reverse_complement(X)     # 16 nt, self-complementary

SPAN_WARNING = ("insert_start/insert_end span the first copy and any copies "
                "fused to it head-to-tail.")


def test_one_copy_of_the_insert_is_not_commented_on():
    product = BACKBONE[:20] + BACKBONE_INS + BACKBONE[20:]
    res = cl.verify_insert(product, BACKBONE_INS, vector_backbone=BACKBONE)
    assert (res["insert_start"], res["insert_end"]) == (20, 48)
    assert res["warnings"] == []


def test_two_copies_fused_head_to_tail_are_reported_as_one_inserted_block():
    product = BACKBONE[:20] + BACKBONE_INS + BACKBONE_INS + BACKBONE[20:]
    assert product.count(BACKBONE_INS) == 2
    res = cl.verify_insert(product, BACKBONE_INS, vector_backbone=BACKBONE)
    assert res["insert_found"] is True
    assert (res["insert_start"], res["insert_end"]) == (20, 76)
    assert res["warnings"] == [
        f"The insert occurs 2 times in the product (at 20, 48). {SPAN_WARNING}"]
    assert not any("vector" in w for w in res["warnings"])


def test_three_copies_extend_the_reported_span_over_the_whole_run():
    res = cl.verify_insert(INSERT * 3, INSERT)
    assert (res["insert_start"], res["insert_end"]) == (0, 96)
    assert res["warnings"] == [
        f"The insert occurs 3 times in the product (at 0, 32, 64). {SPAN_WARNING}"]


def test_a_long_repeat_list_is_truncated_at_eight_positions():
    res = cl.verify_insert(INSERT * 9, INSERT)
    assert res["insert_end"] == 288
    assert res["warnings"] == [
        "The insert occurs 9 times in the product "
        f"(at 0, 32, 64, 96, 128, 160, 192, 224, ...). {SPAN_WARNING}"]


def test_two_copies_that_are_not_adjacent_do_not_merge_into_one_span():
    rc = cl._reverse_complement(INSERT)
    res = cl.verify_insert(INSERT + rc, INSERT)
    assert (res["insert_start"], res["insert_end"]) == (0, 32)
    assert res["warnings"] == [
        "Insert found in BOTH orientations. Verify by restriction mapping.",
        f"The insert occurs 2 times in the product (at 0, 32). {SPAN_WARNING}"]


def test_a_repeated_self_complementary_insert_is_counted_not_double_listed():
    product = "TT" + CLEAN_PALINDROME + CLEAN_PALINDROME + "CC"
    assert cl._occurrences(product, CLEAN_PALINDROME) == [2, 18]
    res = cl.verify_insert(product, CLEAN_PALINDROME)
    assert (res["insert_start"], res["insert_end"]) == (2, 34)
    assert res["warnings"] == [
        "The insert is self-complementary, so its orientation in the product "
        "cannot be distinguished.",
        f"The insert occurs 2 times in the product (at 2, 18). {SPAN_WARNING}"]
