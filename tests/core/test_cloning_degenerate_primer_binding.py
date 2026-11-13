"""Consensus primers are a set of sequences, and `simulate_pcr` read them literally.

A primer written with IUPAC codes (``R`` = A or G) describes the population of
oligos that is actually pipetted.  Matching it with a literal substring test meant
the primer that the lab calls "binds here" was reported as not found in the
template, while its concrete instantiation amplified normally.
"""
import pytest

from biosuite.core import cloning as cl

# 26 nt of real sequence between two poly-T flanks; the EcoRI-ish core carries
# an A at index 5 that a consensus primer would write as R.
TEMPLATE = "TTTTT" + "ATGCCAGCTGCATTGCATCGGATTAC" + "TTTTT"
CONCRETE_FWD = "ATGCCAGCTGCATTGCATCG"
CONSENSUS_FWD = "ATGCC" + "R" + "GCTGCATTGCATCG"
REV = cl._reverse_complement("GATTAC")


def test_the_concrete_primer_anchors_the_expected_product():
    res = cl.simulate_pcr(TEMPLATE, CONCRETE_FWD, REV)
    assert (res["fwd_pos"], res["rev_pos"], res["size"]) == (5, 25, 26)
    assert res["product"] == TEMPLATE[5:31]


def test_the_same_primer_written_as_a_consensus_amplifies_the_same_product():
    assert CONSENSUS_FWD not in TEMPLATE          # absent as a literal string
    res = cl.simulate_pcr(TEMPLATE, CONSENSUS_FWD, REV)
    assert (res["fwd_pos"], res["rev_pos"], res["size"], res["product"]) == \
        (5, 25, 26, TEMPLATE[5:31])


def test_a_reverse_primer_written_as_a_consensus_is_found_through_its_rc():
    # The top-strand binding site is GATTAC; Y = C or T covers the T it holds.
    consensus_site = "GAYTAC"
    assert consensus_site not in TEMPLATE
    res = cl.simulate_pcr(TEMPLATE, CONCRETE_FWD,
                          cl._reverse_complement(consensus_site))
    assert res["rev_pos"] == 25
    assert res["size"] == 26


def test_a_concrete_primer_position_accepts_every_code_that_could_be_that_base():
    assert cl._binding_regex("A").pattern == "(?=[ADHMNRVW])"
    assert cl._binding_regex("R").pattern == "(?=[ABDGHKMNRSVW])"
    assert cl._binding_regex("N").pattern == "(?=[ABCDGHKMNRSTVWY])"


def test_an_ambiguous_template_position_accepts_a_primer_that_shares_a_base_with_it():
    # T G R C G A : the R can be an A, so a plain-A primer binds at 2 as well as 5.
    assert cl._binding_sites("TGRCGA", "A") == [2, 5]
    assert cl._binding_sites("TGRCGA", "Y") == [0, 3]     # T at 0, C at 3
    assert cl._binding_sites("TGRCGA", "N") == [0, 1, 2, 3, 4, 5]


def test_a_code_that_is_not_an_iupac_letter_can_only_match_itself():
    assert cl._binding_sites("AAXAA", "X") == [2]


def test_the_recognition_site_search_stays_literal_about_the_template():
    # Deliberate asymmetry: an enzyme site is a statement about known sequence,
    # so a position whose base is unknown does not complete a site.
    assert cl._find_all("TGRCGA", "A") == [5]


def test_a_consensus_primer_that_is_absent_reports_its_three_prime_anchor():
    anchor = TEMPLATE[10:16]
    primer = "GGGGGG" + anchor[:5] + "N"          # 12 nt, wrong at the 5' end
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(TEMPLATE, primer, REV)
    # The anchor sits at 10, so the full primer would have started at 10 - 6.
    assert str(exc.value) == (
        f"Forward primer '{primer}' not found in template. "
        "Closest partial match (3' 6 bases) at position 4. "
        "Check primer sequence and template.")
