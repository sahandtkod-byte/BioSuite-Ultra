"""The GFF and GTF readers: the nine-column contract and the attribute column.

Both parsers turn one annotation line into one record, and the ninth column is the
only place a gene identifier lives, so an attribute style they fail to read is a
silent loss of the thing the file was written for.
"""
import gzip

import pandas as pd
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

GTF_SPACE_STYLE = (
    'chr1\tsrc\ttranscript\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "t1";'
    ' gene_name "geneA";\n'
    'chr1\tsrc\texon\t100\t150\t.\t+\t.\tgene_id "g1"; transcript_id "t1";'
    ' exon_number "1";\n'
)

GTF_EQUALS_STYLE = (
    "chr1\thavana\ttranscript\t1\t50\t.\t+\t.\tgene_id=ENSG1;transcript_id=ENST1\n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_full_gff3_column_set_becomes_one_typed_record(tmp_path):
    recs = ff.parse_gff(_write(tmp_path, "a.gff", GFF_DEEP))
    assert [(r.seqid, r.feature, r.start, r.end, r.strand, r.phase) for r in recs] == [
        ("chr1", "gene", 100, 200, "+", "."),
        ("chr1", "transcript", 100, 200, "+", "."),
        ("chr1", "exon", 100, 150, "+", "."),
        ("chr1", "exon", 160, 200, "+", "."),
        ("chr1", "CDS", 120, 150, "+", "0")]
    assert recs[0].score == 0 and recs[0].source == "src"
    assert recs[0].attributes == {"ID": "g1", "Name": "geneA",
                                  "gene_type": "protein_coding"}


def test_a_row_with_fewer_than_nine_columns_is_counted_not_silently_lost(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_gff(_write(tmp_path, "a.gff", "chr1\tsrc\tgene\n"
                                                     "chr1\ts\tgene\t10\t50\t.\t+\t.\tID=g1\n"))
    assert [(r.start, r.end, r.attributes) for r in recs] == [(10, 50, {"ID": "g1"})]
    assert "skipped 1 row(s) that were not a usable GFF record" in caplog.text, caplog.text


@pytest.mark.parametrize("text", [
    pytest.param("chr1\tsrc\tgene\tnotanum\t50\t.\t+\t.\tID=g1\n", id="bad-start"),
    pytest.param("chr1\tsrc\tgene\t10\tNaN\t.\t+\t.\tID=g1\n", id="bad-end"),
    pytest.param("chr1\tsrc\tgene\t10\t50\tNA\t+\t.\tID=g1\n", id="bad-score"),
])
def test_a_non_numeric_coordinate_or_score_skips_its_row_and_keeps_the_file(
        tmp_path, caplog, text):
    """One truncated line used to end the whole parse with a bare ValueError.

    The message named neither the file nor the row, and every feature after the
    bad one was never read, which the user saw as an annotation set that had
    shrunk.
    """
    body = text + "chr1\tsrc\texon\t1\t2\t.\t+\t.\tID=ok\n"
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_gff(_write(tmp_path, "bad.gff", body))
    assert [(r.feature, r.attributes) for r in recs] == [("exon", {"ID": "ok"})]
    assert "skipped 1 row(s) that were not a usable GFF record" in caplog.text, caplog.text


def test_comments_and_blank_lines_are_not_counted_as_malformed(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_gff(_write(tmp_path, "a.gff", "##gff-version 3\n\n"
                                                     "#!/bin/sh\n"
                                                     "chr1\ts\tgene\t1\t9\t.\t+\t.\tID=g\n"))
    assert len(recs) == 1
    assert caplog.text == ""


def test_equals_style_and_space_style_attributes_are_both_read(tmp_path):
    rec = ff.parse_gff(_write(tmp_path, "a.gff",
                               "chr1\ts\tgene\t10\t50\t.\t+\t.\tgene foo bar; ID=x\n"))[0]
    assert rec.attributes == {"gene": "foo bar", "ID": "x"}


def test_quotes_around_a_space_style_value_are_not_part_of_the_value(tmp_path):
    rec = ff.parse_gff(_write(tmp_path, "a.gff",
                              'chr1\ts\tgene\t10\t50\t.\t+\t.\tgene_id "ENSG1"\n'))[0]
    assert rec.attributes == {"gene_id": "ENSG1"}


def test_a_lone_token_and_an_empty_item_store_nothing_under_no_key(tmp_path):
    rec = ff.parse_gff(_write(tmp_path, "a.gff",
                              "chr1\ts\tgene\t10\t50\t.\t+\t.\t ID=y ;; Name=Z 1 ;\n"))[0]
    assert rec.attributes == {"ID": "y", "Name": "Z 1"}
    assert "" not in rec.attributes


def test_a_token_with_neither_a_separator_nor_a_value_is_dropped(tmp_path):
    """GFF3 allows a bare flag in the ninth column; it has nothing to store.

    The record still carries the feature, and the flag is not invented into an
    attribute with an empty name — which is what an unstripped '; note' used to do.
    """
    rec = ff.parse_gtf(_write(tmp_path, "a.gtf",
                              'chr1\ts\texon\t5\t30\t.\t+\t.\tgene_id "g1"; loneflag;\n'))[0]
    assert rec.attributes == {"gene_id": "g1"}


def test_a_gtf_row_whose_coordinates_are_not_numbers_skips_only_itself(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_gtf(_write(tmp_path, "b.gtf",
                                   'chr1\ts\texon\tx\t30\t.\t+\t.\tgene_id "g1";\n'
                                   'chr1\ts\texon\t5\t30\t.\t+\t.\tgene_id "g2";\n'))
    assert [r.attributes["gene_id"] for r in recs] == ["g2"]
    assert "skipped 1 row(s) that were not a usable GTF record" in caplog.text, caplog.text


def test_a_gtf_written_with_equals_signs_keeps_its_gene_id(tmp_path):
    """The style GENCODE and several converters emit; reading only spaces lost it.

    The parser this replaces split the ninth column on a space alone, so a record
    carrying gene_id=ENSG1 came back with an empty attribute dict — the identifier
    was gone and nothing said so.
    """
    rec = ff.parse_gtf(_write(tmp_path, "a.gtf", GTF_EQUALS_STYLE))[0]
    assert rec.attributes == {"gene_id": "ENSG1", "transcript_id": "ENST1"}
    assert (rec.feature, rec.start, rec.end) == ("transcript", 1, 50)


def test_a_gtf_written_in_the_classic_style_is_still_read(tmp_path):
    recs = ff.parse_gtf(_write(tmp_path, "a.gtf", GTF_SPACE_STYLE))
    assert [r.attributes for r in recs] == [
        {"gene_id": "g1", "transcript_id": "t1", "gene_name": "geneA"},
        {"gene_id": "g1", "transcript_id": "t1", "exon_number": "1"}]


def test_parse_gff_and_parse_gtf_agree_on_a_file_both_are_pointed_at(tmp_path):
    """The GUI routes .gtf to parse_gff, so the two must not disagree.

    A stricter parser that drops the attributes of a file the looser one reads
    would give two different answers for the same annotation set.
    """
    path = _write(tmp_path, "mixed.gtf", GTF_SPACE_STYLE)
    assert ff.parse_gff(path) == ff.parse_gtf(path)


def test_a_gtf_row_that_is_not_a_feature_is_counted_and_the_file_survives(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_gtf(_write(tmp_path, "a.gtf",
                                   "# a comment line\nchr1\tsrc\n"
                                   "chr1\tsrc\texon\t5\t30\t.\t+\t.\tgene_id \"g1\";\n"))
    assert [(r.feature, r.start, r.score) for r in recs] == [("exon", 5, 0.0)]
    assert "skipped 1 row(s) that were not a usable GTF record" in caplog.text, caplog.text


def test_the_gff_frame_names_attributes_apart_from_the_real_columns(tmp_path):
    path = _write(tmp_path, "hostile.gff",
                  "chr1\tsrc\tgene\t100\t200\t.\t+\t.\tID=g1;start=999;strand=-\n")
    df = ff.gff_to_dataframe(ff.parse_gff(path))
    assert df.loc[0, "start"] == 100 and df.loc[0, "strand"] == "+"
    assert df.loc[0, "info_start"] == "999" and df.loc[0, "info_strand"] == "-"
    assert df.loc[0, "length"] == 101


def test_the_gff_frame_keeps_its_columns_when_there_are_no_records():
    df = ff.gff_to_dataframe([])
    assert list(df.columns) == ["seqid", "source", "feature", "start", "end", "score",
                                "strand", "length"]
    assert df.empty
    pd.testing.assert_series_equal(df["seqid"], pd.Series([], dtype=object, name="seqid"))


def test_the_gff_summary_counts_each_feature_type(tmp_path):
    recs = ff.parse_gff(_write(tmp_path, "a.gff", GFF_DEEP))
    assert ff.format_gff_summary(recs) == ("=== GFF Summary ===\nRecords: 5\n"
                                           "Feature types:\n  exon: 2\n  gene: 1\n"
                                           "  transcript: 1\n  CDS: 1")


def test_the_gff_summary_of_no_records_lists_no_feature_types(tmp_path):
    assert ff.format_gff_summary(ff.parse_gff(_write(tmp_path, "e.gff", "#nothing\n"))) == (
        "=== GFF Summary ===\nRecords: 0")


def test_a_gzipped_gff_is_decompressed_before_parsing(tmp_path):
    path = tmp_path / "a.gff.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("chr1\tsrc\tgene\t100\t200\t.\t+\t.\tID=g1\n")
    assert [r.attributes for r in ff.parse_gff(str(path))] == [{"ID": "g1"}]
