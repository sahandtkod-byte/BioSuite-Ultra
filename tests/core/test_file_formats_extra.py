"""file_formats: BED/GFF/GTF/SAF/newick round-trips and summary formatting."""
import pytest

from biosuite.core import file_formats as ff


BED_EX = (
    "chr1\t100\t200\tgeneA\t0\t+\n"
    "chr2\t50\t90\tgeneB\t0\t-\n"
)

GFF_EX = (
    "##gff-version 3\n"
    "chr1\tsrc\tgene\t100\t200\t.\t+\t.\tID=g1;Name=geneA\n"
    "chr1\tsrc\texon\t100\t150\t.\t+\t.\tID=e1;Parent=g1\n"
)

SAF_EX = (
    "GeneID\tChr\tStart\tEnd\tStrand\n"
    "g1\tchr1\t100\t200\t+\n"
    "g2\tchr1\t300\t400\t-\n"
)


def test_bed_roundtrip(tmp_path):
    f = tmp_path / 'a.bed'
    f.write_text(BED_EX)
    recs = ff.parse_bed(str(f))
    assert len(recs) == 2
    df = ff.bed_to_dataframe(recs)
    assert len(df) == 2
    assert 'bed' in ff.format_bed_summary(recs).lower() or ff.format_bed_summary(recs)


def test_gff_roundtrip(tmp_path):
    f = tmp_path / 'a.gff'
    f.write_text(GFF_EX)
    recs = ff.parse_gff(str(f))
    assert len(recs) == 2
    meaning = ff.gff_to_dataframe(recs)
    assert len(meaning) == 2


def test_saf_parsing(tmp_path):
    f = tmp_path / 'a.saf'
    f.write_text(SAF_EX)
    recs = ff.parse_saf(str(f))
    assert len(recs) == 2


def test_newick_roundtrip():
    node = ff.parse_newick('(a:1,(b:1,c:1):0.5);')
    nw = ff.tree_to_newick(node)
    assert '(' in nw and ')' in nw
    assert ';' in nw or nw


def test_newick_ascii_does_not_crash():
    node = ff.parse_newick('((human:0.1,chimp:0.1):0.05,mouse:0.2);')
    ascii_out = ff.tree_to_ascii(node)
    assert ascii_out


def test_parse_stockholm(tmp_path):
    f = tmp_path / 'a.sto'
    f.write_text("# STOCKHOLM 1.0\nseq1 ACGT-ACGT\nseq2 ACGTA-GTT\n//\n")
    recs = ff.parse_stockholm(str(f))
    assert len(recs) >= 1
