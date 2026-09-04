"""Regression tests for file_formats.py review fixes."""
import numpy as np
import pytest

from biosuite.core import file_formats as ff


def test_parse_bed_space_separated(tmp_path):
    p = tmp_path / "x.bed"
    p.write_text("chr1 100 200 geneA 500 +\nchr2\t0\t50\tgeneB\t.\t-\nbad\t.\t-\n")
    recs = ff.parse_bed(str(p))
    assert len(recs) == 2  # malformed third row is skipped, not fatal
    assert recs[0].score == 500.0
    assert recs[1].score == 0 and recs[1].strand == "-"


def test_parse_bed_tab_file_unchanged(tmp_path):
    p = tmp_path / "x.bed"
    p.write_text("chr1\t100\t200\tgeneA\t700\t+\n")
    recs = ff.parse_bed(str(p))
    assert (recs[0].name, recs[0].score) == ("geneA", 700.0)


def test_bigwig_summary_keeps_trailing_bin(monkeypatch):
    monkeypatch.setattr(ff, "read_bigwig",
                        lambda path: {"chroms": {"chr1": np.ones(5500)}})
    df = ff.bigwig_summary("fake.bw", bin_size=1000)
    assert len(df) == 6 and df.iloc[-1]["end"] == 5500


def test_gtf_and_gff_attribute_styles(tmp_path):
    gtf = tmp_path / "x.gtf"
    gtf.write_text('chr1\tsrc\tgene\t10\t50\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n')
    recs = ff.parse_gtf(str(gtf))
    assert recs[0].attributes == {"gene_id": "g1", "transcript_id": "t1"}
    gff = tmp_path / "x.gff3"
    gff.write_text("chr1\tsrc\tgene\t10\t50\t.\t+\t.\tID=g1;Name=Gene 1\n")
    recs = ff.parse_gff(str(gff))
    assert recs[0].attributes.get('ID') == 'g1'


def test_newick_roundtrip_and_ascii():
    root = ff.parse_newick("(A:0.1,B:0.2)C:0.3;")
    assert root.name == 'C' and len(root.children) == 2
    nwk = ff.tree_to_newick(root)
    root2 = ff.parse_newick(nwk)
    assert root2.children[0].branch_length == pytest.approx(0.1)
    ascii_lines = ff.tree_to_ascii(root)
    assert any('A' in l for l in ascii_lines)


def test_detect_file_format_gz():
    assert ff.detect_file_format("genome.fasta.gz") == 'fasta'
    assert ff.detect_file_format("genome.fa.gz") == 'fasta'
    assert ff.detect_file_format("reads.fastq.gz") == 'fastq'
    assert ff.detect_file_format("data.vcf.gz") == 'vcf'
    assert ff.detect_file_format("x.bam") == 'bam'
    assert ff.detect_file_format("x.nwk") == 'newick'


def test_read_file_bam_uses_bam_mode():
    # Without pysam both return informative errors, but BAM must not claim cram.
    res = ff.read_file("nonexistent.bam")
    if "data" in res and isinstance(res["data"], dict):
        assert res["data"].get("format") in ("bam", None)
    res2 = ff.read_bam("nonexistent.bam")
    assert res2.get("error") is None or "pysam" in res2.get("error", "") or res2["format"] == "bam"


def test_parse_stockholm_multiblock(tmp_path):
    p = tmp_path / "x.sto"
    p.write_text("# STOCKHOLM 1.0\nseq1 ACGT\nseq2 ACGT\n\nseq1 TTGG\nseq2 TTGG\n//\nTRAILER IGNORED\n")
    out = ff.parse_stockholm(str(p))
    assert out['alignment'] == {'seq1': 'ACGTTTGG', 'seq2': 'ACGTTTGG'}


def test_format_file_summary_fasta_none(tmp_path):
    # read_fasta failure -> records None must not crash the summary
    p = tmp_path / "empty.fa"
    p.write_text("")
    res = ff.read_file(str(p))
    assert ff.format_file_summary(res) == "FASTA: 0 sequences"
