"""What `simulate_ligation` accepts as its fragment list.

A bare sequence string is iterable, and every one of its single bases is valid
DNA, so passing one fragment as a string used to "ligate" N 1-nt fragments:
``simulate_ligation("ATGCATGCATGC")`` reported twelve fragments and an efficiency
of 0.1 for a ligation that had never been described.
"""
import pytest

from biosuite.core import cloning as cl

STICKY = "GAATTC" + "ACGT" * 40          # 86 nt, ends in ACGT, starts in GAAT


def test_one_bare_sequence_is_refused_instead_of_being_split_into_bases():
    with pytest.raises(ValueError) as exc:
        cl.simulate_ligation("ATGCATGCATGC")
    assert str(exc.value) == (
        "fragments must be a list of fragment sequences, not one bare sequence "
        "(12 nt was read as 12 1-nt fragments). Pass [sequence] for a single "
        "fragment.")


def test_a_single_fragment_in_a_list_is_ligated_into_a_full_length_product():
    res = cl.simulate_ligation(["ATGCATGC"])
    assert res["product"] == "ATGCATGC"
    assert res["fragment_count"] == 1
    assert res["size"] == 8
    assert res["efficiency"] == 0.95
    assert res["warnings"] == []


def test_a_tuple_of_fragments_is_accepted_like_a_list():
    res = cl.simulate_ligation(("ATGC", "ATGC"))
    assert (res["product"], res["fragment_count"], res["size"]) == ("ATGCATGC", 2, 8)
    assert res["is_circular"] is True
    assert res["efficiency"] == 0.75


def test_no_fragments_is_reported_rather_than_raised():
    res = cl.simulate_ligation([])
    assert res == {
        "product": "", "size": 0, "fragment_count": 0, "is_circular": True,
        "efficiency": 0.0, "verified": False,
        "warnings": ["No fragments provided for ligation."]}


def test_two_sticky_ends_are_joined_head_to_tail_without_a_bonus_they_did_not_earn():
    res = cl.simulate_ligation([STICKY, STICKY])
    assert res["product"] == STICKY + STICKY
    assert res["fragment_count"] == 2
    assert res["size"] == 2 * len(STICKY)
    assert res["warnings"] == []
    # 0.95 base, -0.08 for the second fragment, -0.15 for closing a circle,
    # and no junction bonus: the ACGT end and the GAAT start share no 2-mer.
    assert res["efficiency"] == 0.72


def test_a_linear_ligation_of_the_same_pair_keeps_more_of_its_efficiency():
    res = cl.simulate_ligation([STICKY, STICKY], circular=False)
    assert res["is_circular"] is False
    assert res["efficiency"] == 0.87


def test_the_report_of_a_real_ligation_repeats_the_measured_product():
    text = cl.format_ligation_report(cl.simulate_ligation(["ATCCGG", "AATTCG"]))
    assert "  Fragments ligated: 2" in text
    assert "  Product size: 12 bp" in text
    assert "  Topology: circular" in text
    assert "  Estimated efficiency: 72.0%" in text
    assert "Insert verified" not in text
