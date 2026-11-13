"""Positions returned by the recognition-site search.

These are the plain coordinates every other cloning function is built on, so
they are pinned here with exact answers rather than "not None".
"""
import pytest

from biosuite.core import cloning as cl

ECOARI_TRIPLE = "AAA" + "GAATTC" * 3 + "TTT"


def test_three_tandem_ecori_sites_are_reported_at_their_starts():
    assert cl.find_restriction_sites(ECOARI_TRIPLE, "EcoRI") == [3, 9, 15]


def test_adjacent_sites_start_at_zero_and_at_the_site_length():
    assert cl.find_restriction_sites("GAATTCGAATTC", "EcoRI") == [0, 6]


def test_positions_are_recomputed_from_a_lower_case_input():
    assert cl.find_restriction_sites("aaagaattcctt", "EcoRI") == [3]


def test_overlapping_matches_are_all_returned():
    assert cl._find_all("AAAAAA", "AAA") == [0, 1, 2, 3]
    assert cl._find_all("ATATAT", "ATAT") == [0, 2]


def test_a_site_that_never_occurs_answers_an_empty_list():
    assert cl.find_restriction_sites("ACGT" * 10, "EcoRI") == []


def test_the_search_is_upper_cased_before_the_enzyme_site_is_read():
    assert cl.find_restriction_sites("aAaGgCcTt", "DdeI") == []
    assert cl.find_restriction_sites("AAACTGAGCCC", "DdeI") == [3]


def test_an_enzyme_outside_the_table_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        cl.find_restriction_sites("ACGT", "EcoZZ")
    assert str(exc.value).startswith(
        "Unknown enzyme: EcoZZ. Available: AatII, Acc65I, AflII, ")


def test_a_recognition_site_compiles_to_the_bases_its_codes_stand_for():
    assert cl._site_regex("GAATTC").pattern == "(?=GAATTC)"
    assert cl._site_regex("CTNAG").pattern == "(?=CT[ACGT]AG)"
    assert cl._site_regex("ggatcc").pattern == "(?=GGATCC)"


def test_a_letter_outside_the_iupac_alphabet_compiles_as_itself():
    # Only reachable through a table edit, so it is pinned at the helper.
    assert cl._code_class("A") == "A"
    assert cl._code_class("R") == "[AG]"
    assert cl._code_class("X") == "X"
