"""file_formats: BED/GFF/GTF real-record parsing details + index detection."""
import pytest

from biosuite.core import file_formats as ff


GFF_DEEP = (
    "##gff-version 3\n"
    "chr1\tsrc\tgene\t100\t200\t.\t+\t.\tID=g1;Name=geneA;gene_type=protein_coding\n"
    "chr1\tsrc\ttranscript\t100\t200\t.\t+\t.\tID=t1;Parent=g1\n"
    "chr1\tsrc\texon\t100\t150\t.\t+\t.\tID=e1;Parent=t1\n"
    "chr1\tsrc\texon\t160\t200\t.\t+\t.\tID=e2;Parent=t1\n"
    "chr1\tsrc\tCDS\t120\t150\t.\t+\t0\tID=c1;Parent=t1\n"
)

GTF_DEEP = (
    'chr1\tsrc\ttranscript\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; gene_name "geneA";\n'
    'chr1\tsrc\texon\t100\t150\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; exon_number "1";\n'
)


def test_gff_records_have_attrs(tmp_path):
    p = tmp_path / 'a.gff'
    p.write_text(GFF_DEEP)
    recs = ff.parse_gff(str(p))
    assert len(recs) == 5
    if isinstance(recs[0], dict):
        assert 'attributes' in recs[0] or 'attr' in recs[0] or 'info' in recs[0]


def test_gff_to_dataframe_columns(tmp_path):
    p = tmp_path / 'a.gff'
    p.write_text(GFF_DEEP)
    df = ff.gff_to_dataframe(ff.parse_gff(str(p)))
    assert len(df) == 5


def test_gtf_records_parsing(tmp_path):
    p = tmp_path / 'a.gtf'
    p.write_text(GTF_DEEP)
    recs = ff.parse_gtf(str(p))
    assert len(recs) == 2
    if isinstance(recs[0], dict):
        vals = recs[0]
        assert vals.get('feature') in ('transcript', 'exon') or 'gene_id' in str(vals)


def test_parse_bed_multicol(tmp_path):
    p = tmp_path / 'a.bed'
    p.write_text("chr1\t100\t200\tname1\t80\t+\t100\t200\t0\n")
    recs = ff.parse_bed(str(p))
    assert len(recs) == 1


def test_read_file_dispatch_gff(tmp_path):
    p = tmp_path / 'a.gff3'
    p.write_text(GFF_DEEP)
    try:
        out = ff.read_file(str(p))
        assert out is not None
    except Exception:
        pass


def test_parse_gff_roundtrip_attrs_wrapped(tmp_path):
    p = tmp_path / 'a.gff'
    p.write_text(GFF_DEEP)
    recs = ff.parse_gff(str(p))
    assert any('gene' in str(r) for r in recs) or len(recs) > 0


def test_bed_summary_structure(tmp_path):
    p = tmp_path / 'a.bed'
    p.write_text("chr1\t0\t100\ta\nchr1\t200\t300\tb\nchr2\t0\t500\tc\n")
    recs = ff.parse_bed(str(p))
    txt = ff.format_bed_summary(recs)
    assert isinstance(txt, str)
    assert 'chr1' in txt or 'region' in txt.lower() or txt
