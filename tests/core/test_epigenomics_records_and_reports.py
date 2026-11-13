"""biosuite.core.epigenomics: bisulfite sites, DMRs, ATAC peaks and the report.

Parsing the BED-shaped inputs, analysing methylation, detecting differentially
methylated regions, ATAC peak statistics, the report text, and the pipeline entry
point.
"""
import pytest


@pytest.fixture
def sample_bed(tmp_path):
    """Create a sample bisulfite BED file."""
    bed = tmp_path / "methylation.bed"
    bed.write_text(
        "chr1\t100\t101\t8\t10\tCpG\n"
        "chr1\t200\t201\t3\t10\tCHG\n"
        "chr1\t300\t301\t9\t10\tCHH\n"
        "chr1\t400\t401\t7\t10\tCpG\n"
        "chr1\t500\t501\t2\t10\tCHH\n"
    )
    return str(bed)


class TestEpigenomicsDataclasses:
    def test_methylation_site(self):
        from biosuite.core.epigenomics import MethylationSite
        s = MethylationSite(chrom="chr1", pos=1000, context="CpG",
                            methylation_level=0.8, coverage=20, methylated_count=16)
        assert s.context == "CpG"
        assert s.coverage == 20

    def test_epigenomics_report_defaults(self):
        from biosuite.core.epigenomics import EpigenomicsReport, format_epigenomics_report
        import math
        r = EpigenomicsReport()
        assert r.total_sites == 0
        # None, not 0: a report nobody has run find_dmrs on must not read as
        # "zero differentially methylated regions found".
        assert r.dmr_count is None
        assert r.dmrs == []
        # NaN, not 0.0: no site of that context was measured.
        assert math.isnan(r.cpg_methylation)
        assert math.isnan(r.chg_methylation)
        assert math.isnan(r.chh_methylation)
        assert "DMRs found: not assessed" in format_epigenomics_report(r)
        assert "CpG methylation: n/a (no site of this context)" in format_epigenomics_report(r)


class TestEpigenomicsAnalyzeMethylation:
    def test_parse_bisulfite_bed(self, tmp_path):
        from biosuite.core.epigenomics import parse_bisulfite_bed
        bed = tmp_path / "meth.bed"
        bed.write_text(
            "chr1\t100\t101\t8\t10\tCpG\n"
            "chr1\t200\t201\t3\t10\tCHG\n"
            "chr1\t300\t301\t9\t10\tCHH\n"
            "#comment line\n"
        )
        sites = parse_bisulfite_bed(str(bed))
        assert len(sites) == 3
        assert sites[0].methylation_level == pytest.approx(0.8)

    def test_calculate_methylation_levels(self):
        from biosuite.core.epigenomics import (
            MethylationSite,
            calculate_methylation_levels,
        )
        sites = [
            MethylationSite("chr1", 100, "CpG", 0.9, 20, 18),
            MethylationSite("chr1", 200, "CHG", 0.3, 10, 3),
            MethylationSite("chr1", 300, "CHH", 0.1, 10, 1),
            MethylationSite("chr1", 400, "CpG", 0.85, 20, 17),
        ]
        report = calculate_methylation_levels(sites, min_coverage=5)
        assert report.total_sites == 4
        assert report.avg_methylation > 0
        assert report.cpg_methylation > 0

    def test_calculate_methylation_no_coverage(self):
        from biosuite.core.epigenomics import (
            MethylationSite,
            calculate_methylation_levels,
        )
        sites = [MethylationSite("chr1", 100, "CpG", 0.5, 1, 0)]
        report = calculate_methylation_levels(sites, min_coverage=5)
        assert report.total_sites == 0
        assert "No sites" in report.message


class TestEpigenomicsDetectDMRs:
    def test_find_dmrs(self):
        from biosuite.core.epigenomics import MethylationSite, find_dmrs
        g1 = [
            MethylationSite("chr1", 100, "CpG", 0.9, 30, 27),
            MethylationSite("chr1", 200, "CpG", 0.5, 30, 15),
        ]
        g2 = [
            MethylationSite("chr1", 100, "CpG", 0.1, 30, 3),
            MethylationSite("chr1", 200, "CpG", 0.45, 30, 13),
        ]
        dmrs = find_dmrs(g1, g2, min_coverage=5, p_threshold=0.05, min_delta=0.2)
        assert isinstance(dmrs, list)
        # Position 100 should be a DMR (delta=0.8), position 200 should not (delta=0.05)
        positions = [d['pos'] for d in dmrs]
        assert 100 in positions
        assert 200 not in positions

    def test_find_dmrs_no_common_sites(self):
        from biosuite.core.epigenomics import MethylationSite, find_dmrs
        g1 = [MethylationSite("chr1", 100, "CpG", 0.9, 20, 18)]
        g2 = [MethylationSite("chr1", 500, "CpG", 0.1, 20, 2)]
        dmrs = find_dmrs(g1, g2)
        assert len(dmrs) == 0


class TestEpigenomicsATAC:
    def test_parse_atac_peaks(self, tmp_path):
        from biosuite.core.epigenomics import parse_atac_peaks
        bed = tmp_path / "atac.bed"
        bed.write_text(
            "chr1\t1000\t2000\tpeak1\t10.5\n"
            "chr2\t3000\t4000\tpeak2\t8.0\n"
        )
        peaks = parse_atac_peaks(str(bed))
        assert len(peaks) == 2
        assert peaks[0]['start'] == 1000

    def test_atac_peak_stats(self):
        from biosuite.core.epigenomics import atac_peak_stats
        peaks = [
            {'chrom': 'chr1', 'start': 100, 'end': 400},
            {'chrom': 'chr2', 'start': 500, 'end': 900},
        ]
        stats = atac_peak_stats(peaks)
        assert stats['total_peaks'] == 2
        assert stats['mean_length'] == 350.0

    def test_atac_peak_stats_empty(self):
        from biosuite.core.epigenomics import atac_peak_stats
        assert atac_peak_stats([]) == {}


class TestEpigenomicsFormatReport:
    def test_format_report(self):
        from biosuite.core.epigenomics import (
            EpigenomicsReport,
            format_epigenomics_report,
        )
        r = EpigenomicsReport(total_sites=100, avg_methylation=0.6, cpg_methylation=0.8,
                              chg_methylation=0.3, chh_methylation=0.1, dmr_count=5,
                              methylation_distribution={'unmethylated (< 20%)': 20, 'low (20-50%)': 30,
                                                         'medium (50-80%)': 25, 'high (> 80%)': 25})
        text = format_epigenomics_report(r)
        assert "Epigenomics Report" in text
        assert "100" in text
        assert "CpG methylation" in text


class TestEpigenomicsPipeline:
    """End-to-end epigenomics workflow."""

    def test_bed_to_dmrs(self, sample_bed):
        from biosuite.core.epigenomics import (
            calculate_methylation_levels,
            format_epigenomics_report,
            parse_bisulfite_bed,
        )
        sites = parse_bisulfite_bed(sample_bed)
        assert len(sites) == 5
        report = calculate_methylation_levels(sites, min_coverage=5)
        assert report.total_sites == 5
        text = format_epigenomics_report(report)
        assert "Epigenomics Report" in text
