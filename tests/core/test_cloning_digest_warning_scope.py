"""Which enzymes the two digestion warnings are allowed to name.

Both warnings are about a *table entry*, but the second one describes a cut
position -- and it used to name every non-palindromic enzyme in the request even
when that enzyme found no site at all, so a digest that never cut still claimed a
top-strand approximation.
"""
from biosuite.core import cloning as cl

BSAI_LOWER = "AAAGAGACCTTT"          # GAGACC = the reverse complement of GGTCTC
NO_BSAI_SITE = "AAAA" * 10
BOTH_SITES = "GGATCC" + "C" * 12 + "GAGACC"


def test_a_non_palindromic_enzyme_that_found_no_site_does_not_claim_a_cut_geometry():
    res = cl.simulate_digestion(NO_BSAI_SITE, "BsaI", topology="linear")
    assert res["cuts"] == []
    assert res["warnings"] == [
        "cut geometry derived from unchecked table entries for: BsaI"]


def test_a_non_palindromic_enzyme_that_did_cut_still_reports_the_approximation():
    res = cl.simulate_digestion(BSAI_LOWER, "BsaI", topology="linear")
    assert len(res["cuts"]) == 1
    assert res["warnings"] == [
        "cut geometry derived from unchecked table entries for: BsaI",
        "non-palindromic site recognised on either strand, but the double-strand "
        "cut geometry of such enzymes is not modelled, so cut_pos is a top-strand "
        "approximation for: BsaI",
    ]


def test_the_geometry_warning_names_only_the_enzyme_that_actually_cut():
    res = cl.simulate_digestion("GGATCC" + "C" * 20, ["BamHI", "BsaI"],
                                topology="linear")
    assert [c["enzyme"] for c in res["cuts"]] == ["BamHI"]
    # BamHI is REBASE-verified and palindromic, and BsaI found no site at all:
    # the cut-geometry warning may name only an enzyme that produced a cut, while
    # the table-entry caveat is about BsaI's unverified data wherever it came from.
    assert res["warnings"] == [
        "cut geometry derived from unchecked table entries for: BsaI"]
    assert not any("not modelled" in w for w in res["warnings"])


def test_the_unverified_entry_warning_lists_every_affected_enzyme_in_order():
    res = cl.simulate_digestion(BSAI_LOWER + "GAAGAC" + "GGTCTC" + "A" * 8,
                               ["BsaI", "BbsI", "EcoRI"], topology="linear")
    assert res["warnings"] == [
        "cut geometry derived from unchecked table entries for: BbsI, BsaI",
        "non-palindromic site recognised on either strand, but the double-strand "
        "cut geometry of such enzymes is not modelled, so cut_pos is a top-strand "
        "approximation for: BbsI, BsaI",
    ]


def test_a_verified_palindromic_enzyme_cuts_without_any_warning():
    res = cl.simulate_digestion("AAAGAATTC" + "C" * 30, "EcoRI", topology="linear")
    assert res["warnings"] == []
    assert res["overhangs"] == {"EcoRI": "5prime"}
