"""Parameter refusals in crispr.py and the empty-matrix edges in popgen.py.

Every number below was read off the running code.  Before the change:
`design_guides(seq, pam_type='SpCA9')` designed SpCas9 guides and announced
"Found 20 guide RNAs for SpCA9"; `max_guides=-5` was a slice and returned 34
guides; `guide_length=0` died with ZeroDivisionError; `tool='crispor2'` ran the
built-in designer without a word; `nucleotide_diversity` over zero sites returned
NaN behind a numpy warning; `linkage_disequilibrium(max_pairs=0)` raised
ZeroDivisionError and `max_pairs=-5` raised "cannot convert float NaN to integer";
`calculate_fst` against an empty population answered FST 0.0; and `full_analysis`
on a site-less matrix raised IndexError.
"""
import math
import re

import numpy as np
import pytest

from biosuite.core import crispr as cr
from biosuite.core import popgen as pg

_rng = np.random.default_rng(7)
SEQ = "".join(np.array(list("ACGT"))[_rng.integers(0, 4, 400)])


# ── crispr: the designer's own parameters ────────────────────────────────────

def test_the_builtin_designer_finds_the_pam_sites_it_claims():
    result = cr.design_guides(SEQ)
    assert (result.engine, result.num_guides) == ("builtin", 20)
    assert result.message == "Found 20 guide RNAs for SpCas9"


@pytest.mark.parametrize("pam_type, expected", [("SpCas9", 39), ("SaCas9", 13),
                                                ("Cas12a", 12), ("SpCas9-HF1", 39)])
def test_each_nuclease_uses_its_own_pattern(pam_type, expected):
    """The four patterns are not interchangeable — a typo used to select NGG while
    the message echoed the requested name.  HF1 shares SpCas9's NGG site, which is
    why both count 39 here."""
    assert cr.design_guides(SEQ, pam_type=pam_type, max_guides=500).num_guides == expected


@pytest.mark.parametrize("pam_type", ["SpCA9", "spcas9", "Cas12", "", None])
def test_an_unknown_nuclease_is_refused_by_name(pam_type):
    with pytest.raises(ValueError) as exc:
        cr.design_guides(SEQ, pam_type=pam_type)
    assert str(exc.value) == (
        f"Unknown pam_type: {pam_type!r}. Available: SpCas9, SaCas9, Cas12a, "
        "SpCas9-HF1.")


@pytest.mark.parametrize("bad, rendered", [(0, "0"), (-3, "-3"), (2.5, "2.5"),
                                            (True, "True")])
def test_a_useless_guide_length_is_refused_not_divided_by(bad, rendered):
    with pytest.raises(ValueError,
                       match=rf"guide_length must be a positive integer, got "
                             rf"{re.escape(rendered)}\."):
        cr.design_guides(SEQ, guide_length=bad)


@pytest.mark.parametrize("bad, rendered", [(0, "0"), (-5, "-5"), (4.5, "4.5")])
def test_a_useless_guide_cap_is_refused_not_sliced(bad, rendered):
    """max_guides=-5 used to return *more* guides than the default cap: 34 of the
    39 candidate sites, because [-5:] of a sorted list is a slice, not a limit."""
    with pytest.raises(ValueError,
                       match=rf"max_guides must be a positive integer, got "
                             rf"{re.escape(rendered)}\."):
        cr.design_guides(SEQ, max_guides=bad)


def test_the_guide_cap_still_caps():
    assert cr.design_guides(SEQ, max_guides=5).num_guides == 5
    assert cr.design_guides(SEQ, max_guides=1).num_guides == 1


@pytest.mark.parametrize("tool", ["crispor2", "CRISPOR", "", None, "local"])
def test_an_unknown_tool_is_refused_rather_than_running_the_builtin_one(tool):
    with pytest.raises(ValueError,
                       match=rf"^Unknown tool: {re.escape(str(tool))}\. "
                              rf"Use 'auto' or 'crispor'\.$"):
        cr.design_guides(SEQ, tool=tool)


def test_the_two_real_tools_still_work(monkeypatch):
    """'crispor' keeps its documented behaviour: it warns and designs locally,
    because the API result parsing is not implemented.  The request itself is
    stubbed — this suite never touches the network."""
    called = []
    monkeypatch.setattr(cr, "_crispor_api_search",
                        lambda seq: called.append(seq) or None)
    assert cr.design_guides(SEQ, tool="auto").engine == "builtin"
    assert called == [], "the built-in path must not contact the API"
    assert cr.design_guides(SEQ, tool="crispor").engine == "builtin"
    assert called == [SEQ]


def test_an_ungsearched_off_target_field_says_nothing():
    guides = cr.design_guides(SEQ).guides
    assert {g.off_target_count for g in guides} == {None}, \
        "0 would read as 'this guide is specific'"
    text = cr.format_crispr_report(cr.design_guides(SEQ))
    assert "Off-target" not in text, "the report never showed the field"
    assert "Found 20 guide RNAs for SpCas9" in text


def test_an_empty_target_is_still_a_plain_answer():
    result = cr.design_guides("", pam_type="nonsense")
    assert (result.engine, result.message) == ('none', "No sequence provided")


# ── popgen: matrices with nothing in them ────────────────────────────────────

def test_pi_over_no_sites_is_zero_not_nan():
    got = pg.nucleotide_diversity(np.zeros((3, 0)))
    assert got == 0.0
    assert not math.isnan(got)


def test_pi_over_a_real_matrix_is_unchanged():
    """Positive control: the empty guard must not swallow the calculation."""
    gm = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    freqs = gm.mean(axis=0) / 2
    per_site = 2 * freqs * (1 - freqs) * 3 / 2
    assert pg.nucleotide_diversity(gm) == pytest.approx(float(per_site.mean()))


@pytest.mark.parametrize("bad", [0, -5, 2.5, None])
def test_a_useless_pair_budget_is_refused(bad):
    gm = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError,
                       match=rf"max_pairs must be an integer >= 1, got {re.escape(str(bad))}\."):
        pg.linkage_disequilibrium(gm, max_pairs=bad)


def test_the_pair_budget_still_limits():
    rng = np.random.default_rng(11)
    gm = rng.integers(0, 3, (6, 40)).astype(float)
    assert len(pg.linkage_disequilibrium(gm, max_pairs=3)) == 3
    assert pg.linkage_disequilibrium(gm, max_pairs=1000), "the default still computes"


def test_fst_against_an_empty_population_is_refused():
    """An empty sample has no allele frequency; the NaN arithmetic collapsed the
    pair to FST 0.0, which reads as 'these two populations do not differ'."""
    with pytest.raises(ValueError,
                       match=r"population 1 has no samples\."):
        pg.calculate_fst([np.zeros((2, 3)), np.zeros((0, 3))])


def test_fst_of_a_single_population_is_an_empty_answer():
    assert pg.calculate_fst([np.array([[0.0, 1.0]])]) == {}


def test_full_analysis_without_sites_answers_instead_of_crashing():
    report = pg.full_analysis(np.zeros((4, 0)))
    assert report.num_sites == 0 and report.hw_test == {}
    assert report.message == ("No sites in the genotype matrix, so no per-site "
                              "statistic was computed.")
    assert report.nucleotide_diversity == 0.0 and report.tajima_d == 0.0
    assert pg.format_popgen_report(report).startswith("=== Population Genetics Report ===")


def test_full_analysis_on_a_real_matrix_still_fills_every_field():
    rng = np.random.default_rng(2)
    gm = rng.integers(0, 3, (12, 8)).astype(float)
    labels = np.array(["a"] * 6 + ["b"] * 6)
    report = pg.full_analysis(gm, population_labels=labels)
    assert report.num_sites == 8 and report.num_populations == 2
    assert set(report.hw_test) >= {"chi2", "p_value", "in_hwe"}
    assert report.ld and list(report.fst) == [(0, 1)]
    assert 0.0 <= report.fst[(0, 1)] <= 1.0
    assert "FST (Nei's Gst)" in pg.format_popgen_report(report)
