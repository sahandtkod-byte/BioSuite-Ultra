"""Bisulfite BED and ATAC peak parsing: the records a real pipeline emits that
must be skipped rather than turned into wrong numbers.

`parse_bisulfite_bed` reads six fixed fields and derives the methylation fraction
from columns 4 and 5; `parse_atac_peaks` reads a BED with optional score and name
columns.  Both had only seen well-formed lines in tests, so the short-record and
non-numeric paths — the ones that would otherwise produce a site with a bogus
level or crash the whole file — were unexecuted.  `find_dmrs` is pushed into the
case where the methylation difference clears `min_delta` but Fisher's exact test
says the counts cannot support it, which is the difference between "no DMR" and a
false positive in the report.
"""
import math

import pytest

from biosuite.core.epigenomics import (
    MethylationSite,
    calculate_methylation_levels,
    find_dmrs,
    parse_atac_peaks,
    parse_bisulfite_bed,
)


def bed_line(chrom, start, end, meth, total, context=None):
    fields = [chrom, str(start), str(end), str(meth), str(total)]
    if context is not None:
        fields.append(context)
    return "\t".join(fields) + "\n"


# ── bisulfite BED ────────────────────────────────────────────────────────────

def test_a_complete_record_is_parsed_with_its_context(tmp_path):
    path = tmp_path / "calls.bed"
    path.write_text(bed_line("chr1", 100, 101, 8, 10, "CpG"), encoding="utf-8")
    site = parse_bisulfite_bed(str(path))[0]
    assert (site.chrom, site.pos, site.context) == ("chr1", 100, "CpG")
    assert site.methylated_count == 8 and site.coverage == 10
    assert site.methylation_level == pytest.approx(0.8)


def test_a_record_short_of_the_two_count_columns_is_skipped(tmp_path):
    """Four fields cannot yield a methylation fraction; guessing one would invent
    a site that no read supported."""
    path = tmp_path / "calls.bed"
    path.write_text("chr1\t100\t101\t8\n" + bed_line("chr1", 200, 201, 1, 4, "CpG"),
                    encoding="utf-8")
    sites = parse_bisulfite_bed(str(path))
    assert [s.pos for s in sites] == [200]


def test_comments_and_track_lines_are_not_counted_as_data(tmp_path):
    path = tmp_path / "calls.bed"
    path.write_text("# browser guide line\ntrack type=bed\n"
                    + bed_line("chr1", 5, 6, 2, 2, "CpG"), encoding="utf-8")
    assert [s.pos for s in parse_bisulfite_bed(str(path))] == [5]


def test_a_non_numeric_count_column_skips_the_row_not_the_file(tmp_path):
    path = tmp_path / "calls.bed"
    path.write_text("chr1\t10\t11\tNA\t20\tCpG\n"
                    "chr1\t20\t21\t5\tten\tCpG\n"
                    + bed_line("chr1", 30, 31, 6, 6, "CpG"), encoding="utf-8")
    sites = parse_bisulfite_bed(str(path))
    assert [s.pos for s in sites] == [30]


def test_a_zero_coverage_row_is_not_a_division_error(tmp_path):
    """max(int(parts[4]), 1) keeps the fraction at 0.0 for an uncovered site."""
    path = tmp_path / "calls.bed"
    path.write_text(bed_line("chr1", 0, 1, 0, 0, "CpG"), encoding="utf-8")
    site = parse_bisulfite_bed(str(path))[0]
    assert site.coverage == 0
    assert site.methylation_level == 0.0


def test_an_empty_file_yields_no_sites(tmp_path):
    path = tmp_path / "empty.bed"
    path.write_text("", encoding="utf-8")
    assert parse_bisulfite_bed(str(path)) == []


# ── DMR detection ────────────────────────────────────────────────────────────

def site(chrom, pos, methylated, total):
    return MethylationSite(chrom=chrom, pos=pos, context="CpG",
                           methylation_level=methylated / max(total, 1),
                           coverage=total, methylated_count=methylated)


def test_a_difference_without_statistical_support_is_not_called_a_dmr():
    """Deep coverage on both sides: the 0.25 delta clears min_delta but Fisher's
    exact test has plenty of counts to explain it away, so no DMR may be reported.
    """
    g1 = [site("chr1", 100, 25, 100)]
    g2 = [site("chr1", 100, 0, 100)]
    dmrs = find_dmrs(g1, g2, min_coverage=5, p_threshold=0.05, min_delta=0.2)
    assert dmrs, "a 25%% vs 0%% difference at 100x is significant"
    assert dmrs[0]["p_value"] < 0.05

    weak1 = [site("chr1", 200, 3, 4)]
    weak2 = [site("chr1", 200, 1, 4)]
    assert find_dmrs(weak1, weak2, min_coverage=2, p_threshold=0.05,
                     min_delta=0.2) == [], "one read either way cannot support a DMR"


def test_a_site_below_min_coverage_is_excluded_from_both_sides():
    g1 = [site("chr1", 100, 10, 10)]
    g2 = [site("chr1", 100, 0, 2)]
    assert find_dmrs(g1, g2, min_coverage=5) == []


def test_positions_present_in_one_group_only_are_ignored():
    g1 = [site("chr1", 1, 10, 10)]
    g2 = [site("chr1", 2, 0, 10)]
    assert find_dmrs(g1, g2, min_coverage=5) == []


def test_a_clear_dmr_carries_both_levels_and_the_rounded_statistics():
    g1 = [site("chr1", 500, 50, 50)]
    g2 = [site("chr1", 500, 0, 50)]
    dmrs = find_dmrs(g1, g2, min_coverage=5, p_threshold=0.05, min_delta=0.2)
    assert len(dmrs) == 1
    dmr = dmrs[0]
    assert dmr["chrom"] == "chr1" and dmr["pos"] == 500
    assert dmr["delta_methylation"] == pytest.approx(1.0)
    assert dmr["group1_level"] == pytest.approx(1.0)
    assert dmr["group2_level"] == pytest.approx(0.0)
    assert 0.0 <= dmr["p_value"] < 1e-6


def test_the_smaller_difference_is_reported_first():
    """Sorted by position, not by effect size — pinned because the report reads
    the list in order."""
    g1 = [site("chr1", 10, 10, 10), site("chr1", 5, 6, 10)]
    g2 = [site("chr1", 10, 0, 10), site("chr1", 5, 0, 10)]
    dmrs = find_dmrs(g1, g2, min_coverage=5, min_delta=0.2)
    assert [d["pos"] for d in dmrs] == [5, 10]


def test_calculate_methylation_levels_reports_only_the_covered_sites():
    sites = [site("chr1", 1, 5, 5), site("chr1", 2, 1, 1)]
    report = calculate_methylation_levels(sites, min_coverage=5)
    assert report.total_sites == 1, "the 1-read site is not counted"
    assert report.avg_methylation == pytest.approx(1.0)
    assert report.cpg_methylation == pytest.approx(1.0)
    assert math.isnan(report.chg_methylation) and math.isnan(report.chh_methylation), \
        "a context with no sites is not measured at 0.0"
    assert report.methylation_distribution["high (> 80%)"] == 1
    assert report.methylation_distribution["unmethylated (< 20%)"] == 0


def test_a_sample_where_no_site_has_enough_coverage_says_so():
    report = calculate_methylation_levels([site("chr1", 1, 1, 2)], min_coverage=5)
    assert report.total_sites == 0
    assert report.message == "No sites passed coverage filter"


def test_each_context_is_averaged_separately():
    sites = [site("chr1", 1, 10, 10), site("chr1", 2, 0, 10),
             MethylationSite(chrom="chr1", pos=3, context="CHG",
                             methylation_level=0.5, coverage=10, methylated_count=5)]
    report = calculate_methylation_levels(sites, min_coverage=5)
    assert report.total_sites == 3
    assert report.cpg_methylation == pytest.approx(0.5)
    assert report.chg_methylation == pytest.approx(0.5)
    assert report.avg_methylation == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert report.methylation_distribution == {
        "unmethylated (< 20%)": 1, "low (20-50%)": 0,
        "medium (50-80%)": 1, "high (> 80%)": 1}


# ── ATAC peaks ───────────────────────────────────────────────────────────────

def test_a_three_column_bed_gets_zero_score_and_empty_name(tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text("chr1\t100\t200\n", encoding="utf-8")
    peak = parse_atac_peaks(str(path))[0]
    assert peak == {"chrom": "chr1", "start": 100, "end": 200, "score": 0, "name": ""}


def test_a_five_column_bed_carries_name_and_score(tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text("chr9\t10\t300\tpeak_a\t47.5\n", encoding="utf-8")
    peak = parse_atac_peaks(str(path))[0]
    assert peak["name"] == "peak_a" and peak["score"] == pytest.approx(47.5)
    assert peak["end"] - peak["start"] == 290


def test_comment_lines_and_short_records_are_skipped(tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text("# a comment\nchr1\t100\nchr2\t1\t2\tname\tscore\n"
                    "chr3\t5\t9\n", encoding="utf-8")
    peaks = parse_atac_peaks(str(path))
    assert [p["chrom"] for p in peaks] == ["chr3"]


def test_a_bed_of_only_comments_has_no_peaks(tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text("# nothing here\n# still nothing\n", encoding="utf-8")
    assert parse_atac_peaks(str(path)) == []
