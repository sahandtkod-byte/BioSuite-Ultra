"""The GWAS text report: which sentences it is allowed to print.

`format_gwas_report(results)` is the one-argument call — the lead list is optional —
and it used to print ``No significant loci detected.`` for it.  A study with two
SNPs below 5e-8 was therefore reported as a study with no loci whenever the caller
left the second argument out; the same line was also the honest answer after a lead
scan that really had found nothing.
"""
import pandas as pd
import pytest

from biosuite.core.gwas import detect_lead_snps, format_gwas_report, run_gwas

SIGNIFICANT = pd.DataFrame([
    {"chrom": "chr1", "pos": 1000, "snp_id": "rs1",
     "case_alt": 90, "case_ref": 10, "ctrl_alt": 10, "ctrl_ref": 90},
    {"chrom": "chr1", "pos": 2000, "snp_id": "rs2",
     "case_alt": 85, "case_ref": 15, "ctrl_alt": 15, "ctrl_ref": 85},
])

RESULTS = pd.DataFrame([
    {"snp_id": "rs-strong", "chrom": "chr1", "pos": 100, "p_value": 1e-40},
    {"snp_id": "rs-suggestive", "chrom": "chr1", "pos": 200, "p_value": 1e-6},
    {"snp_id": "rs-null", "chrom": "chr1", "pos": 300, "p_value": 0.5},
])


def test_the_header_and_its_three_counts_are_fixed_layout():
    report = format_gwas_report(RESULTS, lead_snps=None)
    lines = report.splitlines()

    assert lines[0] == "GWAS Analysis Report", lines
    assert lines[1] == "=" * 40, lines
    assert lines[2] == "Total SNPs tested: 3", lines
    assert lines[3] == "Genome-wide significant (p < 5e-8): 1", lines
    assert lines[4] == "Suggestive (p < 1e-5): 2", lines
    assert lines[5] == "", lines


def test_the_suggestive_line_counts_the_stricter_band_separately():
    """A SNP at 1e-6 is suggestive and not genome-wide; the two numbers must differ."""
    genome_wide = int((RESULTS["p_value"] < 5e-8).sum())
    suggestive = int((RESULTS["p_value"] < 1e-5).sum())

    assert suggestive == 2 and genome_wide == 1
    report = format_gwas_report(RESULTS, lead_snps=None)
    assert f"Genome-wide significant (p < 5e-8): {genome_wide}" in report
    assert f"Suggestive (p < 1e-5): {suggestive}" in report


def test_an_omitted_lead_list_admits_that_no_scan_ran():
    report = format_gwas_report(run_gwas(SIGNIFICANT))

    assert "Lead SNP detection was not run, so no loci are listed." in report, report
    assert "No significant loci detected." not in report, report


def test_a_scan_that_found_nothing_says_so_exactly():
    results = run_gwas(SIGNIFICANT)
    nothing = detect_lead_snps(results, p_threshold=1e-300)

    report = format_gwas_report(results, nothing)

    assert "No significant loci detected." in report, report
    assert "was not run" not in report, report


def test_a_lead_line_names_the_snp_its_position_its_p_and_its_odds_ratio():
    results = run_gwas(SIGNIFICANT)
    leads = detect_lead_snps(results, window_kb=0)

    lines = format_gwas_report(results, leads).splitlines()

    assert lines[6] == "Lead SNPs:", lines
    assert lines[7] == "  rs1 (chr1:1000) p=5.57e-29 OR=81.00", lines[7]
    assert lines[8] == "  rs2 (chr1:2000) p=1.70e-22 OR=32.11", lines[8]
    assert len(lines) == 9, lines


def test_every_lead_is_listed_rather_than_the_first_few():
    leads = pd.DataFrame([
        {"snp_id": f"rs{i}", "chrom": "chr1", "pos": i, "p_value": 1e-20,
         "odds_ratio": 1.5} for i in range(9)
    ])

    report = format_gwas_report(RESULTS, leads)

    assert report.count("\n  rs") == 9, report


def test_a_run_with_no_testable_snp_reports_zeroes_and_no_loci():
    nothing_tested = run_gwas(pd.DataFrame([
        {"chrom": "chr1", "pos": 100, "snp_id": "rs",
         "case_alt": 0, "case_ref": 0, "ctrl_alt": 0, "ctrl_ref": 0}]))

    report = format_gwas_report(nothing_tested, detect_lead_snps(nothing_tested))

    assert "Total SNPs tested: 0" in report, report
    assert "Genome-wide significant (p < 5e-8): 0" in report, report
    assert "No significant loci detected." in report, report


@pytest.mark.parametrize("p_value, genome_wide, suggestive", [
    (5e-8, 0, 1), (4.9e-8, 1, 1), (1e-5, 0, 0), (9.9e-6, 0, 1),
])
def test_the_band_boundaries_are_exclusive_as_the_labels_promise(p_value, genome_wide,
                                                                suggestive):
    frame = pd.DataFrame([{"snp_id": "rs", "chrom": "chr1", "pos": 1,
                           "p_value": p_value, "odds_ratio": 1.0}])

    report = format_gwas_report(frame, lead_snps=None)

    assert f"Genome-wide significant (p < 5e-8): {genome_wide}" in report, report
    assert f"Suggestive (p < 1e-5): {suggestive}" in report, report
