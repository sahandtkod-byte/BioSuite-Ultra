"""The BED reader: which rows become records, and which are refused.

parse_bed's output is the record count a user reads back as the contents of their
file, so every row the parser declines to read has to be counted and said out loud,
and an interval the format does not allow must not be turned into a negative length.
"""
import gzip

import pandas as pd
import pytest

from biosuite.core import file_formats as ff


def _bed(tmp_path, text, name="regions.bed"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_bed3_row_yields_the_three_columns_the_format_promises(tmp_path):
    recs = ff.parse_bed(_bed(tmp_path, "chr1\t1000\t2000\n"))
    assert [(r.chrom, r.start, r.end) for r in recs] == [("chr1", 1000, 2000)]
    assert (recs[0].name, recs[0].score, recs[0].strand) == ("", 0, ".")


def test_a_bed6_row_fills_name_score_and_strand(tmp_path):
    rec = ff.parse_bed(_bed(tmp_path, "chr2\t50\t90\tgeneB\t10\t-\n"))[0]
    assert (rec.chrom, rec.start, rec.end, rec.name, rec.score, rec.strand) == (
        "chr2", 50, 90, "geneB", 10.0, "-")


def test_columns_seven_to_nine_become_the_thick_and_rgb_fields(tmp_path):
    rec = ff.parse_bed(_bed(tmp_path, "chr1\t100\t200\tg\t5\t+\t120\t180\t255,0,0\n"))[0]
    assert (rec.thick_start, rec.thick_end, rec.item_rgb) == (120, 180, "255,0,0")


def test_the_bed12_block_columns_have_no_place_in_a_record_and_are_dropped(tmp_path):
    """BED12 is read as BED9, which is what the docstring now says.

    blockCount, blockSizes and blockSaved (columns 10-12) would need their own
    fields; a caller that reads the support claim as exon geometry would get
    nothing, so the claim and the code have to agree.
    """
    rec = ff.parse_bed(_bed(tmp_path, "chr1\t100\t200\tg\t5\t+\t120\t180\t255,0,0"
                                      "\t3\t1,1\t0,50\n"))[0]
    assert (rec.item_rgb, rec.thick_start, rec.thick_end) == ("255,0,0", 120, 180)
    assert not hasattr(rec, "block_count") and not hasattr(rec, "block_sizes")
    assert len(vars(rec)) == 9


def test_a_dot_in_the_score_column_is_no_score_rather_than_a_parse_error(tmp_path):
    rec = ff.parse_bed(_bed(tmp_path, "chr2\t0\t50\tgeneB\t.\t-\n"))[0]
    assert (rec.score, rec.strand) == (0, "-")


def test_a_row_separated_by_spaces_is_read_like_a_tab_separated_one(tmp_path):
    recs = ff.parse_bed(_bed(tmp_path, "chr1 100 200 geneA 500 +\n"))
    assert [(r.chrom, r.start, r.end, r.name, r.score, r.strand) for r in recs] == [
        ("chr1", 100, 200, "geneA", 500.0, "+")]


@pytest.mark.parametrize("line", ["", "\t", "# a comment", "track name=t", "browser position chr1:1-10"])
def test_comments_and_track_and_browser_lines_are_not_records_or_malformed_rows(
        tmp_path, caplog, line):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, line + "\nchr1\t100\t200\tg\n"))
    assert [(r.chrom, r.start, r.end) for r in recs] == [("chr1", 100, 200)]
    assert "skipped" not in caplog.text, caplog.text


def test_a_file_of_only_comments_reads_as_zero_records_and_says_nothing(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        assert ff.parse_bed(_bed(tmp_path, "# one\ntrack name=t\n")) == []
    assert caplog.text == ""


def test_a_row_with_fewer_than_three_columns_is_counted_not_silently_lost(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, "chr1\t100\nchr1\t1\t2\nchromX\t0\t10\n"))
    assert [(r.chrom, r.start, r.end) for r in recs] == [("chr1", 1, 2), ("chromX", 0, 10)]
    assert "skipped 1 row(s) that were not a usable BED record" in caplog.text, caplog.text


def test_a_coordinate_that_is_not_a_number_is_counted_with_the_rest_of_the_file(
        tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, "chr1\tx\t200\nchr1\t1\t2\nchr1\t3\tbad\n"))
    assert [(r.start, r.end) for r in recs] == [(1, 2)]
    assert "skipped 2 row(s)" in caplog.text, caplog.text


def test_a_non_numeric_score_drops_that_interval_and_says_so(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, "chr1\t100\t200\tg\t*\t+\n"))
    assert recs == []
    assert "skipped 1 row(s) that were not a usable BED record" in caplog.text, caplog.text


@pytest.mark.parametrize("line,why", [
    ("chr1\t300\t100\tx\n", "an end before its start"),
    ("chr1\t-5\t10\tx\n", "a negative start"),
])
def test_an_interval_the_format_does_not_allow_is_refused(tmp_path, caplog, line, why):
    """Kept, such a row subtracts from the total the summary reports.

    A 300-100 interval has length -200, so one typo in a browser-extended track
    quietly shrinks 'Total bp' instead of being reported as unusable.
    """
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, line))
    assert recs == [], why
    assert "skipped 1 row(s) that were not a usable BED record" in caplog.text, caplog.text


def test_a_zero_length_interval_is_a_record_and_not_malformed(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        recs = ff.parse_bed(_bed(tmp_path, "chr1\t100\t100\tpoint\n"))
    assert [(r.start, r.end) for r in recs] == [(100, 100)]
    assert caplog.text == ""


def test_a_blank_row_does_not_stop_the_rows_after_it(tmp_path):
    recs = ff.parse_bed(_bed(tmp_path, "chr1\t100\t200\tg1\n\n\nchr1\t300\t400\tg2\n"))
    assert [r.name for r in recs] == ["g1", "g2"]


def test_a_gzipped_bed_is_decompressed_before_parsing(tmp_path):
    path = tmp_path / "regions.bed.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("chr1\t100\t200\tg1\n")
    recs = ff.parse_bed(str(path))
    assert [(r.chrom, r.start, r.end, r.name) for r in recs] == [("chr1", 100, 200, "g1")]


def test_a_missing_bed_file_raises_naming_the_path(tmp_path):
    missing = str(tmp_path / "gone.bed")
    with pytest.raises(FileNotFoundError) as exc:
        ff.parse_bed(missing)
    assert "gone.bed" in str(exc.value)


def test_the_bed_summary_counts_records_chromosomes_and_base_pairs(tmp_path):
    recs = ff.parse_bed(_bed(tmp_path, "chr1\t0\t100\ta\nchr1\t200\t300\tb\nchr2\t0\t500\tc\n"))
    assert ff.format_bed_summary(recs) == ("=== BED File Summary ===\nRecords: 3\n"
                                          "Chromosomes: 2\nTotal bp: 700")


def test_the_bed_summary_of_no_records_reports_nothing_as_zero(tmp_path):
    assert ff.format_bed_summary(ff.parse_bed(_bed(tmp_path, "# only a header\n"))) == (
        "=== BED File Summary ===\nRecords: 0")


def test_the_bed_frame_keeps_its_columns_when_there_are_no_records():
    df = ff.bed_to_dataframe([])
    assert list(df.columns) == ["chrom", "start", "end", "name", "score", "strand", "length"]
    assert df.empty
    pd.testing.assert_series_equal(df["chrom"], pd.Series([], dtype=object, name="chrom"))


def test_the_bed_frame_lengths_are_half_open_bed_intervals(tmp_path):
    df = ff.bed_to_dataframe(ff.parse_bed(_bed(tmp_path, "chr1\t100\t200\ta\n")))
    assert df.to_dict("records") == [{"chrom": "chr1", "start": 100, "end": 200,
                                     "name": "a", "score": 0, "strand": ".", "length": 100}]
