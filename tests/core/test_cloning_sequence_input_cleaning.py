"""`_validate_dna` claims to clean a sequence; only its ends used to be cleaned.

A sequence pasted out of a FASTA record arrives in lines, and the line breaks
came back as ``Invalid characters in <role>:`` followed by an invisible newline,
so the same molecule read differently depending on how it was typed in.
"""
import pytest

from biosuite.core import cloning as cl

MULTI_LINE = "ATGCATGC\nATGCATGC\r\n\tAAAA\n"
ONE_LINE = "ATGCATGCATGCATGCAAAA"


def test_a_multi_line_sequence_reads_as_the_same_molecule_as_the_one_line_form():
    assert cl._validate_dna(MULTI_LINE, "t") == ONE_LINE


def test_a_sequence_of_only_whitespace_is_refused_as_empty_by_role():
    with pytest.raises(ValueError) as exc:
        cl._validate_dna("\n \t \r\n", "digestion template")
    assert str(exc.value) == "digestion template is empty"


def test_lowercase_is_upper_cased_and_the_iupac_alphabet_is_accepted():
    assert cl._validate_dna("ACGTNacgtn", "t") == "ACGTNACGTN"


@pytest.mark.parametrize("bad", ["ACGTX", "ACGT-P", "ATGCÜ"])
def test_a_character_outside_the_alphabet_is_refused_sorted_and_once(bad):
    with pytest.raises(ValueError) as exc:
        cl._validate_dna(bad, "t")
    reported = str(exc.value).split(": ")[1].split(".")[0]
    assert reported == "".join(sorted(set(bad) - set("ATCGWSMKRYBDHVN")))


# ── every public entry point sees the cleaned sequence ───────────────────────

def test_the_site_search_finds_a_site_whose_letters_are_split_across_lines():
    assert cl.find_restriction_sites("GAAT\nTC", enzyme="EcoRI") == [0]


def test_a_digest_of_a_wrapped_sequence_cuts_where_the_plain_sequence_is_cut():
    wrapped = "TTTT\nGAATTC\n" + "A" * 20 + "\n"
    plain = "TTTT" + "GAATTC" + "A" * 20
    assert (cl.simulate_digestion(wrapped, "EcoRI", topology="linear")["sizes"]
            == cl.simulate_digestion(plain, "EcoRI", topology="linear")["sizes"]
            == [5, 25])


def test_pcr_amplifies_a_template_that_was_pasted_with_line_breaks():
    res = cl.simulate_pcr("ATGCATGC\nATGCATGC", "ATGC", "GCAT")
    assert (res["fwd_pos"], res["rev_pos"], res["size"]) == (0, 4, 8)


def test_primers_are_designed_from_the_cleaned_template():
    pair = cl.design_primers("ATGC\nATGC", primer_length=4)
    assert (pair["forward"], pair["reverse"]) == ("ATGC", "GCAT")


def test_a_ligated_fragment_list_is_cleaned_fragment_by_fragment():
    res = cl.simulate_ligation(["ATGC\n", "GG\nGG"])
    assert res["product"] == "ATGCGGGG"
    assert res["fragment_count"] == 2


def test_an_insert_written_on_two_lines_is_found_in_the_product():
    res = cl.verify_insert("TTTT" + ONE_LINE + "CCCC", "ATGCATGC\nATGCATGCAAAA")
    assert res["insert_found"] is True
    assert (res["insert_start"], res["insert_end"]) == (4, 24)
