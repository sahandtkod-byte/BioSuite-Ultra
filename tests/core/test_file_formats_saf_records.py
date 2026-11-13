"""The SAF reader: featureCounts' annotation summary, column by column.

A SAF row is a gene and an interval, and the strand column is optional, so what
the file does not say must not be filled in as if it did.
"""
import pytest

from biosuite.core import file_formats as ff


def _write(tmp_path, text, name="counts.saf"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_full_featureCounts_row_becomes_five_fields(tmp_path):
    recs = ff.parse_saf(_write(tmp_path, "GeneID\tChr\tStart\tEnd\tStrand\n"
                                         "g1\tchr1\t10\t200\t-\n"))
    assert recs == [{"gene_id": "g1", "chr": "chr1", "start": 10, "end": 200,
                     "strand": "-"}]


def test_the_header_row_is_not_a_record_and_is_not_counted_as_broken(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_saf(_write(tmp_path, "GeneID\tChr\tStart\tEnd\tStrand\n"
                                             "g1\tchr1\t10\t20\t+\n"))
    assert [r["gene_id"] for r in recs] == ["g1"]
    assert caplog.text == ""


def test_a_row_written_without_a_strand_column_says_so_instead_of_guessing(tmp_path):
    """'.' is what an absent strand is; '+' would be a claim the file never made.

    Strand-specific counts and their opposites are pooled by strand, so a gene
    reported as '+' because the column was missing is sorted into the wrong set.
    """
    recs = ff.parse_saf(_write(tmp_path, "g1\tchr1\t10\t200\n"))
    assert recs == [{"gene_id": "g1", "chr": "chr1", "start": 10, "end": 200,
                     "strand": "."}]


def test_a_row_with_fewer_than_four_columns_is_counted(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_saf(_write(tmp_path, "short\trow\n"
                                             "g1\tchr1\t10\t20\t+\n"))
    assert len(recs) == 1
    assert "skipped 1 row(s) that were not a usable SAF record" in caplog.text, caplog.text


def test_a_non_numeric_coordinate_is_counted_and_the_file_survives(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_saf(_write(tmp_path, "g3\tchr2\tNaN\t20\t.\n"
                                             "g4\tchr2\t30\t40\t-\n"))
    assert [r["gene_id"] for r in recs] == ["g4"]
    assert "skipped 1 row(s)" in caplog.text, caplog.text


def test_comments_and_blank_lines_are_neither_records_nor_losses(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_saf(_write(tmp_path, "# featureCounts output\n\n"
                                             "g1\tchr1\t1\t2\t+\n"))
    assert len(recs) == 1
    assert caplog.text == ""


def test_a_start_larger_than_the_end_is_read_as_written(tmp_path):
    """SAF coordinates are copied, not repaired — only BED intervals are refused.

    The two formats number differently (BED is half-open from 0, SAF is the
    featureCounts interval), so this pins that no silent normalising happens here.
    """
    recs = ff.parse_saf(_write(tmp_path, "g1\tchr1\t200\t100\t+\n"))
    assert (recs[0]["start"], recs[0]["end"]) == (200, 100)


def test_an_empty_annotation_file_reads_as_no_features(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        assert ff.parse_saf(_write(tmp_path, "")) == []
    assert caplog.text == ""


def test_a_missing_saf_file_raises_naming_the_path(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        ff.parse_saf(str(tmp_path / "gone.saf"))
    assert "gone.saf" in str(exc.value)


def test_the_saf_summary_counts_the_features_it_read(tmp_path):
    result = {"format": "saf", "records": ff.parse_saf(
        _write(tmp_path, "g1\tchr1\t1\t2\t+\ng2\tchr1\t3\t4\t-"))}
    assert ff.format_file_summary(result) == "SAF: 2 annotated features"
