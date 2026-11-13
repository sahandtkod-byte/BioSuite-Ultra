"""Contexts that were never measured, and the BED row that broke the DMR scan.

`calculate_methylation_levels` averaged `contexts.get('CpG', [0])`, so a CHH-only
dataset reported `CpG methylation: 0.000` — a claim that cytosines in that context
are unmethylated when no such site existed.  `dmr_count` defaulted to 0 and printed
`DMRs found: 0` for a report nobody had run `find_dmrs` on.  A BED row whose
methylated count exceeded its coverage reached Fisher's exact test as a negative
cell and aborted the whole scan with
`ValueError: All values in `table` must be nonnegative`.
"""
import logging
import math

import pytest

from biosuite.core.epigenomics import (
    EpigenomicsReport,
    MethylationSite,
    calculate_methylation_levels,
    format_epigenomics_report,
    parse_bisulfite_bed,
)


def site(context, level, coverage=10, methylated=None):
    return MethylationSite(chrom="chr1", pos=1, context=context,
                           methylation_level=level, coverage=coverage,
                           methylated_count=coverage * level if methylated is None
                           else methylated)


def write_bed(tmp_path, rows):
    path = tmp_path / "meth.bed"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_an_unmeasured_context_is_nan_not_zero():
    report = calculate_methylation_levels([site("CHH", 0.5)], min_coverage=1)
    assert report.cpg_methylation != report.cpg_methylation, "NaN, and not equal to itself"
    assert math.isnan(report.chg_methylation)
    assert report.chh_methylation == pytest.approx(0.5)
    assert report.avg_methylation == pytest.approx(0.5)


def test_every_context_that_was_measured_keeps_its_number():
    report = calculate_methylation_levels(
        [site("CpG", 0.8), site("CHG", 0.2), site("CHH", 0.6)], min_coverage=1)
    assert report.cpg_methylation == pytest.approx(0.8)
    assert report.chg_methylation == pytest.approx(0.2)
    assert report.chh_methylation == pytest.approx(0.6)


def test_the_report_says_which_contexts_were_not_measured():
    text = format_epigenomics_report(
        calculate_methylation_levels([site("CHH", 0.5)], min_coverage=1))
    assert "CpG methylation: n/a (no site of this context)" in text
    assert "CHG methylation: n/a (no site of this context)" in text
    assert "CHH methylation: 0.500" in text
    assert "DMRs found: not assessed" in text


def test_a_report_with_a_recorded_dmr_count_still_prints_it():
    """Positive control: the honest default must not hide a real number."""
    text = format_epigenomics_report(EpigenomicsReport(
        total_sites=4, avg_methylation=0.25, cpg_methylation=0.25,
        chg_methylation=0.25, chh_methylation=0.25, dmr_count=5))
    assert "DMRs found: 5" in text
    assert "not assessed" not in text
    assert "n/a" not in text


def test_a_methylated_count_above_the_coverage_is_skipped_with_a_warning(
        tmp_path, caplog):
    """The row parses fine, so nothing upstream rejected it, and one such line
    stopped the whole DMR scan."""
    bed = write_bed(tmp_path, [
        "chr1\t100\t101\t8\t10\tCpG",
        "chr1\t200\t201\t14\t10\tCpG",      # more methylated reads than there are
        "chr1\t300\t301\t2\t4\tCHH",
    ])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.epigenomics"):
        sites = parse_bisulfite_bed(bed)
    assert [(s.pos, s.methylation_level) for s in sites] == [(100, 0.8), (300, 0.5)]
    assert any("14 methylated reads of 10 total" in r.getMessage() for r in caplog.records)


def test_an_unparseable_row_is_named_by_line_number(tmp_path, caplog):
    bed = write_bed(tmp_path, ["chr1\t100\t101\t8\t10\tCpG",
                               "chr1\tstart\t101\t8\t10\tCpG"])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.epigenomics"):
        sites = parse_bisulfite_bed(bed)
    assert [s.pos for s in sites] == [100]
    assert any("line 2" in r.getMessage() for r in caplog.records)


def test_the_context_column_is_always_taken_from_the_row(tmp_path):
    """The `else 'CpG'` fallback sat under a `len(parts) < 6: continue` guard, so
    it could never be reached; the column is the only source of the context."""
    bed = write_bed(tmp_path, ["chr1\t5\t6\t1\t2\tCHG"])
    sites = parse_bisulfite_bed(bed)
    assert [(s.context, s.methylation_level) for s in sites] == [("CHG", 0.5)]


def test_a_short_row_is_skipped_without_a_warning(tmp_path):
    bed = write_bed(tmp_path, ["chr1\t5\t6", "chr1\t5\t6\t1\t2"])
    assert parse_bisulfite_bed(bed) == []
