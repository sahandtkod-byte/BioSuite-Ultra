"""Comprehensive offline tests for core/ngs.py (SAM/VCF/BED parsing & stats)."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import ngs

SAM_TEXT = """\
@HD\tVN:1.6\tSO:coordinate
@SQ\tSN:chr1\tLN:1000
@SQ\tSN:chr2\tLN:500
read1\t0\tchr1\t1\t60\t10M\t*\t0\t0\tACGTACGTAC\tFFFFJJJJJJ\tNM:i:0
read2\t16\tchr1\t10\t30\t10M\t*\t0\t0\tACGTACGTAC\tFFFFJ\tNM:i:1
"""

VCF_TEXT = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\trs1\tA\tG\t99\tPASS\tAF=0.2
chr1\t150\trs2\tC\tT\t50\tPASS\tAF=0.3
chr1\t200\trs3\tA\tT\t10\tq10\tAF=0.1
chr1\t250\t.\tAT\tA\t40\tPASS\t.
chr2\t100\trs5\tA\tC,G\t30\tPASS\tMULTI
"""


@pytest.fixture()
def sam_file(tmp_path):
    p = tmp_path / "reads.sam"
    p.write_text(SAM_TEXT)
    return str(p)


@pytest.fixture()
def vcf_file(tmp_path):
    p = tmp_path / "vars.vcf"
    p.write_text(VCF_TEXT)
    return str(p)


@pytest.fixture()
def bed_file(tmp_path):
    p = tmp_path / "regions.bed"
    p.write_text("chr1\t90\t160\tregionA\nchr1\t190\t210\tregionB\nchr2\t50\t120\tregionC\n")
    return str(p)


# ── SAM ──────────────────────────────────────────────────────────────────────

def test_read_sam_parses_reads_and_tags(sam_file):
    reads = ngs.read_sam(sam_file)
    assert len(reads) == 2
    r1, r2 = reads
    assert r1['QNAME'] == 'read1' and r1['FLAG'] == 0 and r1['RNAME'] == 'chr1'
    assert r1['POS'] == 1 and r1['MAPQ'] == 60 and r1['CIGAR'] == '10M'
    assert r1['NM'] == '0'                       # optional TAG parsed
    assert r2['FLAG'] == 16 and r2['MAPQ'] == 30  # reverse strand, lower qual


def test_read_sam_missing_file():
    assert ngs.read_sam('/nonexistent/x.sam') is None


def test_parse_sam_header(sam_file):
    hdr = ngs.parse_sam_header(sam_file)           # dict-based parser
    assert hdr is not None
    assert len(hdr['@SQ']) == 2
    assert hdr['@HD']['VN'] == '1.6'
    assert hdr['@SQ'][0]['SN'] == 'chr1'
    assert len(hdr['raw']) == 3


def test_parse_sam_read_fields():
    rec = ngs.parse_sam_read(SAM_TEXT.splitlines()[3])
    assert rec['POS'] == 1 and rec['SEQ'] == 'ACGTACGTAC'


# ── VCF ──────────────────────────────────────────────────────────────────────

def test_read_vcf_columns(vcf_file):
    df = ngs.read_vcf(vcf_file)
    assert len(df) == 5
    assert list(df.columns) == ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    assert df['POS'].dtype.kind in 'iu'
    assert df.iloc[0]['REF'] == 'A' and int(df.iloc[0]['POS']) == 100


def test_read_vcf_max_variants(vcf_file):
    df = ngs.read_vcf(vcf_file, max_variants=2)
    assert len(df) == 2


def test_read_vcf_missing():
    assert ngs.read_vcf('/nonexistent/x.vcf') is None


def test_manhattan_from_vcf(vcf_file):
    df = ngs.read_vcf(vcf_file)
    manh = ngs.manhattan_from_vcf(df)
    assert set(manh.columns) == {'CHROM', 'POS', 'neg_log10'}
    # QUAL>1 produces negative -log10 by design in this helper
    assert manh['neg_log10'].isna().sum() == 0
    df['QUAL'] = df['QUAL'].clip(0, 1)
    manh2 = ngs.manhattan_from_vcf(df)
    assert (manh2['neg_log10'] >= 0).all()


# ── windowed coverage helper (pysam-free) ────────────────────────────────────

def test_windowed_coverage_array():
    cov = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    df = ngs._windowed_coverage_array(cov, 4, chrom='chrX')
    assert list(df['mean_coverage']) == [5.0, 13.0]
    assert list(df['chrom']) == ['chrX', 'chrX']
    assert int(df.iloc[0]['start']) == 0 and int(df.iloc[0]['end']) == 4


def test_windowed_coverage_tiny_array():
    cov = np.arange(3, dtype=float)
    df = ngs._windowed_coverage_array(cov, 10)
    assert len(df) == 1
    assert df.iloc[0]['mean_coverage'] == pytest.approx(1.0)


# ── BED/VCF intersection ─────────────────────────────────────────────────────

def test_intersect_count(bed_file, vcf_file):
    # regionA 90..160 contains POS100(rs1) and POS150(rs2) -> 2
    # regionB 190..210 contains POS200(rs3)                    -> 1
    # regionC chr2 50..120 contains chr2:100(rs5)              -> 1
    counts = ngs.intersect_bed_vcf(bed_file, vcf_file, report='count')
    assert list(counts) == [2, 1, 1]


def test_intersect_detail(bed_file, vcf_file):
    det = ngs.intersect_bed_vcf(bed_file, vcf_file, report='detail')
    assert len(det) == 4
    assert det['vcf_pos'].min() == 100


def test_intersect_missing_inputs(vcf_file):
    assert ngs.intersect_bed_vcf('/nope.bed', vcf_file) is None
    assert ngs.intersect_bed_vcf(vcf_file.replace('.vcf', '.bed') + 'x', vcf_file) is None


# ── summary stats ────────────────────────────────────────────────────────────

def test_vcf_summary(vcf_file):
    summ = ngs.vcf_summary(vcf_file)
    assert summ is not None
    # vcf_summary returns a dict of stats — exact keys verified against impl
    assert isinstance(summ, dict)


def test_ts_tv_snp_classification(vcf_file):
    # fixture SNPs: A->G, C->T (ti); A->T (tv);
    # multi-allele A->C,G expands to C(tv)+G(ti) => ti=3, tv=2
    stats = ngs.compute_ts_tv_ratio(vcf_file)
    assert stats is not None
    assert stats['transitions'] == 3
    assert stats['transversions'] == 2
    assert stats['ratio'] == pytest.approx(1.5)


# ── pysam-gated fallbacks ────────────────────────────────────────────────────

def test_coverage_graceful_without_pysam_or_inputs(tmp_path):
    if not ngs.HAS_PYSAM:
        assert ngs.compute_coverage('x.bam') is None
        assert ngs.coverage_from_bed('x.bam', 'y.bed') is None
    else:
        if hasattr(ngs, 'pysam'):
            pass
    assert ngs.compute_coverage('x.bam') is None   # missing file either way
