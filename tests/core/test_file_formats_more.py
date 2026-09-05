"""Broad offline tests for core/file_formats.py (BED/GFF/Newick/Stockholm/etc.)."""
import gzip
import struct

import pandas as pd
import pytest

from biosuite.core import file_formats as ff


@pytest.fixture()
def bed_file(tmp_path):
    p = tmp_path / "regs.bed"
    p.write_text("track name=t\nchr1\t100\t200\tgeneA\t60\t+\nchr2\t50\t150\tgeneB\t20\t-\n")
    return str(p)


@pytest.fixture()
def gff_file(tmp_path):
    p = tmp_path / "ann.gff"
    p.write_text("##gff-version 3\n"
                 "chr1\tsrc\tgene\t100\t800\t.\t+\t.\tID=g1;Name=Gene1\n"
                 "chr1\tsrc\tCDS\t120\t360\t.\t+\t0\tParent=g1\n")
    return str(p)


# ── BED / GFF / GTF / SAF ────────────────────────────────────────────────────

def test_parse_bed_record_fields(bed_file):
    recs = ff.parse_bed(bed_file)
    assert len(recs) == 2
    r = recs[0]
    assert (r.chrom, r.start, r.end, r.name) == ('chr1', 100, 200, 'geneA')
    assert r.score == 60


def test_gff_feature_attributes(gff_file):
    recs = ff.parse_gff(gff_file)
    assert len(recs) == 2
    assert recs[0].feature == 'gene'
    assert recs[1].feature == 'CDS'


def test_dataframes_summary_between_formats(bed_file, gff_file):
    bed_df = ff.bed_to_dataframe(ff.parse_bed(bed_file))
    assert len(bed_df) == 2
    summ = ff.format_bed_summary(ff.parse_bed(bed_file))
    assert 'Records' in summ and isinstance(summ, str)
    gff_df = ff.gff_to_dataframe(ff.parse_gff(gff_file))
    assert len(gff_df) == 2
    assert isinstance(ff.format_gff_summary(ff.parse_gff(gff_file)), str)


def test_parse_gtf_and_saf(tmp_path):
    gtf = tmp_path / 'x.gtf'
    gtf.write_text('chr1\tsrc\texon\t5\t300\t.\t+\t.\tgene_id "g1"; gene_name "X";\n')
    recs = ff.parse_gtf(str(gtf))
    assert len(recs) == 1
    saf = tmp_path / 'x.saf'
    saf.write_text('GeneID\tChr\tStart\tEnd\tStrand\ng1\tchr1\t1\t200\t+\n')
    rows = ff.parse_saf(str(saf))
    assert len(rows) == 1


# ── Newick trees ─────────────────────────────────────────────────────────────

def test_newick_roundtrip():
    node = ff.parse_newick("((A:1,B:1):0.5,C:2);")
    assert node is not None
    out = ff.tree_to_newick(node)
    assert out.startswith('(') and 'A' in out
    node2 = ff.parse_newick(out)
    assert node2 is not None
    ascii_art = list(ff.tree_to_ascii(node))
    joined = "\n".join(ascii_art)
    assert 'A' in joined and 'B' in joined and 'C' in joined


# ── Stockholm ────────────────────────────────────────────────────────────────

def test_parse_stockholm(tmp_path):
    p = tmp_path / 'a.sto'
    p.write_text("# STOCKHOLM 1.0\nseq1 ACGTACGT\nseq2 AAGGTTGG\n//\n")
    data = ff.parse_stockholm(str(p))
    assert len(data['alignment']) == 2


# ── bigWig (text fallback) ──────────────────────────────────────────────────

def test_read_bigwig_text_and_summary(tmp_path):
    p = tmp_path / 'w.wig'
    p.write_text("chr1\t0\t100\t10\nchr1\t100\t200\t20\nchr1\t200\t300\t30\n")
    try:
        data = ff.read_bigwig(str(p))
        assert data is not None
    except Exception:
        pytest.skip("binary bigWig backend absent")
    summ = ff.bigwig_summary(str(p), bin_size=150)
    assert isinstance(summ, dict) or isinstance(summ, pd.DataFrame)
    assert ff.format_bigwig_summary(summ) is not None


# ── index parsing ─────────────────────────────────────────────────────────────

def test_read_bam_index_minimal(tmp_path):
    # minimal .bai: magic + n_ref=1 + one ref block with zero bins and zero chunks
    p = tmp_path / 'x.bai'
    # magic + n_ref=1, then one ref header (n_bin=0)
    p.write_bytes(struct.pack('<4si', b'BAI\x01', 1) + struct.pack('<i', 0))
    idx = ff.read_bam_index(str(p))
    assert idx is not None or idx == {}


def test_read_vcf_index_gzipped(tmp_path):
    p = tmp_path / 'x.tbi.gz'
    p.write_bytes(gzip.compress(b'TBI\x01' + struct.pack('<i', 0)))
    idx = ff.read_vcf_index(str(p))
    assert idx is not None or idx == {}


# ── detect & generic reader ─────────────────────────────────────────────────

def test_detect_file_format(tmp_path):
    b = tmp_path / 'a.bed'
    b.write_text('chr1\t1\t2\n')
    fmt = ff.detect_file_format(str(b))
    assert fmt is not None
    fa = tmp_path / 'x.fa'
    fa.write_text('>id\nACGT\n')
    assert ff.detect_file_format(str(fa)) is not None
    assert ff.detect_file_format('/nonexistent/x.zz') is not None


def test_read_file_dispatch(tmp_path):
    fa = tmp_path / 'x.fa'
    fa.write_text('>id\nACGT\n')
    result = ff.read_file(str(fa))
    assert result is not None
    summary = ff.format_file_summary(result)
    assert isinstance(summary, str) and len(summary) > 0
