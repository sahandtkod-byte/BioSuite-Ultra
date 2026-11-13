"""The columns each GWAS function reads by name, and what happens when one is missing.

`run_gwas` grew a schema check because an untestable run used to return an empty
frame *with no columns*, which killed the next function in the chain with a
KeyError.  The same reasoning applies to the two functions downstream of it: they
read `snp_id`, `p_value` and `odds_ratio` out of whatever frame they are handed,
including frames a caller built by hand.
"""
import pandas as pd
import pytest

from biosuite.core.gwas import (
    GWAS_RESULT_COLUMNS, detect_lead_snps, format_gwas_report, run_gwas,
)

SIGNIFICANT = pd.DataFrame([
    {"chrom": "chr1", "pos": 1000, "snp_id": "rs1",
     "case_alt": 90, "case_ref": 10, "ctrl_alt": 10, "ctrl_ref": 90},
    {"chrom": "chr1", "pos": 2000, "snp_id": "rs2",
     "case_alt": 85, "case_ref": 15, "ctrl_alt": 15, "ctrl_ref": 85},
])


# ── run_gwas ────────────────────────────────────────────────────────────────

def test_missing_count_columns_are_reported_with_the_documented_names():
    with pytest.raises(ValueError) as exc:
        run_gwas(pd.DataFrame([{"chrom": "chr1", "pos": 1}]))

    message = str(exc.value)
    assert "missing required column" in message
    assert "case_alt, case_ref, ctrl_alt, ctrl_ref" in message


def test_position_columns_are_reported_as_missing_too():
    counts_only = pd.DataFrame([{"case_alt": 1, "case_ref": 1,
                                 "ctrl_alt": 1, "ctrl_ref": 1}])

    with pytest.raises(ValueError, match=r"missing required column\(s\): chrom, pos"):
        run_gwas(counts_only)


def test_non_default_prefixes_name_both_accepted_columns():
    with pytest.raises(ValueError) as exc:
        run_gwas(pd.DataFrame([{"chrom": "chr1", "pos": 1}]),
                 case_col="tumor", control_col="healthy")

    message = str(exc.value)
    assert "tumor_alt (or case_alt)" in message
    assert "healthy_ref (or ctrl_ref)" in message


def test_documented_schema_and_alternative_prefixes_are_both_accepted():
    """The GUI help and the docstring promise ``ctrl_*``; custom prefixes work too."""
    renamed = SIGNIFICANT.rename(columns={
        "case_alt": "tumor_alt", "case_ref": "tumor_ref",
        "ctrl_alt": "healthy_alt", "ctrl_ref": "healthy_ref"})

    default_names = run_gwas(SIGNIFICANT)
    prefixed = run_gwas(renamed, case_col="tumor", control_col="healthy")

    assert list(prefixed["snp_id"]) == list(default_names["snp_id"])
    assert prefixed["p_value"].tolist() == pytest.approx(default_names["p_value"].tolist())


def test_a_frame_with_no_rows_at_all_is_a_missing_column_not_an_empty_result():
    """``pd.DataFrame()`` has no columns, so the schema error is the answer."""
    with pytest.raises(ValueError, match="missing required column"):
        run_gwas(pd.DataFrame())


# ── detect_lead_snps ────────────────────────────────────────────────────────

def test_the_lead_scan_asks_for_the_identifier_its_docstring_promised():
    """It sorted, compared and de-duplicated on ``snp_id`` while documenting only
    chrom, pos and p_value, so the frame it described could not be used."""
    minimal = pd.DataFrame([{"chrom": "chr1", "pos": 1000, "p_value": 1e-9}])

    with pytest.raises(ValueError) as exc:
        detect_lead_snps(minimal)

    assert "needs column(s) snp_id" in str(exc.value), str(exc.value)
    assert "the frame carries chrom, pos, p_value" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("missing", ["chrom", "pos", "p_value"])
def test_every_column_the_lead_scan_reads_is_named_when_it_is_absent(missing):
    frame = pd.DataFrame([{"chrom": "chr1", "pos": 1, "snp_id": "rs", "p_value": 1e-9}])
    without = frame.drop(columns=[missing])

    with pytest.raises(ValueError, match=f"needs column\\(s\\) {missing}"):
        detect_lead_snps(without)


def test_a_run_that_tested_nothing_can_still_be_handed_to_the_lead_scan():
    untestable = pd.DataFrame([{"chrom": "chr1", "pos": 100, "snp_id": "rs",
                               "case_alt": 0, "case_ref": 0,
                               "ctrl_alt": 0, "ctrl_ref": 0}])

    results = run_gwas(untestable)
    leads = detect_lead_snps(results)

    assert list(results.columns) == list(GWAS_RESULT_COLUMNS)
    assert leads.empty and list(leads.columns) == list(GWAS_RESULT_COLUMNS)


# ── format_gwas_report ──────────────────────────────────────────────────────

def test_the_report_names_the_column_it_needed_for_the_counts():
    with pytest.raises(ValueError, match="needs column\\(s\\) p_value"):
        format_gwas_report(pd.DataFrame([{"chrom": "chr1"}]))


def test_the_report_names_the_column_it_needed_for_a_lead_line():
    """A lead list assembled by hand can omit ``odds_ratio``, and a KeyError about
    it names neither the frame nor the function that wanted it."""
    hand_made_leads = pd.DataFrame([{"chrom": "chr1", "pos": 1, "snp_id": "rs",
                                     "p_value": 1e-9}])

    with pytest.raises(ValueError, match="needs column\\(s\\) odds_ratio"):
        format_gwas_report(run_gwas(SIGNIFICANT), hand_made_leads)


def test_an_empty_lead_frame_needs_no_columns_to_be_reported_as_empty():
    results = run_gwas(SIGNIFICANT)
    empty_leads = results.iloc[0:0]

    assert "No significant loci detected." in format_gwas_report(results, empty_leads)
