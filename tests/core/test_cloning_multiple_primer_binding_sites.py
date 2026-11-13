"""A primer that matches twice: which pair defines the product.

`simulate_pcr` read only the first occurrence of each primer, so a template
carrying the reverse-primer binding site anywhere upstream of the forward site
was rejected as an orientation error even though the downstream copy would have
amplified normally.  The product is now the first forward site that has a reverse
site downstream of it, with the nearest such site.
"""
import pytest

from biosuite.core import cloning as cl

FWD = "AAGGTCCA"                    # 8 nt
REV_SITE = "TTCAGG"                 # top-strand site the reverse primer reads
REV = cl._reverse_complement(REV_SITE)

# One decoy reverse site upstream of the forward primer, the real one downstream.
DECOY_UPSTREAM = REV_SITE + "C" * 20 + FWD + "G" * 20 + REV_SITE + "T" * 10
# Two forward sites, one reverse site between them.
TWO_FORWARD = FWD + "A" * 10 + REV_SITE + "C" * 10 + FWD + "G" * 10
# Two reverse sites, both upstream of the only forward site.
NO_VALID_PAIR = "A" * 10 + REV_SITE + "C" * 10 + REV_SITE + "G" * 10 + FWD + "T" * 5


def test_the_setups_mean_what_they_say():
    assert cl._binding_sites(DECOY_UPSTREAM, FWD) == [26]
    assert cl._binding_sites(DECOY_UPSTREAM, REV_SITE) == [0, 54]
    assert cl._binding_sites(TWO_FORWARD, FWD) == [0, 34]
    assert cl._binding_sites(TWO_FORWARD, REV_SITE) == [18]
    assert cl._binding_sites(NO_VALID_PAIR, FWD) == [42]
    assert cl._binding_sites(NO_VALID_PAIR, REV_SITE) == [10, 26]


def test_a_decoy_reverse_site_upstream_no_longer_refuses_the_reaction():
    res = cl.simulate_pcr(DECOY_UPSTREAM, FWD, REV)
    assert (res["fwd_pos"], res["rev_pos"], res["size"]) == (26, 54, 34)
    assert res["product"] == DECOY_UPSTREAM[26:60]


def test_the_leftmost_forward_site_that_has_a_partner_defines_the_product():
    res = cl.simulate_pcr(TWO_FORWARD, FWD, REV)
    assert (res["fwd_pos"], res["rev_pos"], res["size"]) == (0, 18, 24)
    assert res["product"] == TWO_FORWARD[:24]


def test_a_pair_with_no_forward_first_ordering_is_still_refused_by_both_positions():
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(NO_VALID_PAIR, FWD, REV)
    assert str(exc.value) == (
        "Primer orientation error: forward primer binds at position 42 "
        "but reverse primer RC binds at position 10. Forward primer must bind "
        "upstream of the reverse primer binding site.")


def test_two_primers_that_share_one_base_position_are_refused():
    template = "ATGCATGC" + "TTTTTTTT"
    primer = "ATGCATGC"
    assert cl._binding_sites(template, primer) == [0]
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(template, primer, cl._reverse_complement(primer))
    assert str(exc.value) == (
        "Primer orientation error: forward primer binds at position 0 "
        "but reverse primer RC binds at position 0. Forward primer must bind "
        "upstream of the reverse primer binding site.")


def test_a_forward_primer_that_is_absent_still_names_itself_and_not_a_position():
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr("TTTT" + REV_SITE + "A" * 20, FWD, REV)
    assert str(exc.value) == (
        f"Forward primer '{FWD}' not found in template. "
        "Check primer sequence and template.")
