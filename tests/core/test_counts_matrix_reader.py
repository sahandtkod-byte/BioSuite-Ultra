"""`read_counts_matrix` — delimiter sniffing, shifted headers, and refusals.

The reader has to cope with whatever a pipeline dumped: TSV, CSV, semicolon or
space separated, comment lines, and the two header conventions that differ only
in whether the header carries a leading separator.  A malformed matrix is worse
than a missing one, because the caller sees a tidy DataFrame and analyses the
wrong column as expression, so the tests below pin both the recovered frames and
the cases where the function must return ``None``.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.expression import read_counts_matrix


@pytest.fixture()
def write_matrix(tmp_path):
    """Drop a text blob into a named file and hand back its path."""
    def _write(text, name="matrix.txt"):
        path = tmp_path / name
        path.write_text(text, newline="")
        return str(path)
    return _write


# ── the formats the docstring promises ───────────────────────────────────────

def test_tab_separated_matrix_renamed_to_gene_plus_samples(write_matrix):
    df = read_counts_matrix(write_matrix("gene\ts1\ts2\nA\t1\t2\nB\t3\t4\n"))
    assert list(df.columns) == ["gene", "Sample_1", "Sample_2"]
    assert df["gene"].tolist() == ["A", "B"]
    assert df[["Sample_1", "Sample_2"]].values.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_comma_separated_matrix(write_matrix):
    df = read_counts_matrix(write_matrix("gene,s1,s2\nA,1,2\nB,3,4\n"))
    assert df["gene"].tolist() == ["A", "B"]
    assert df["Sample_2"].tolist() == [2.0, 4.0]


def test_semicolon_separated_matrix(write_matrix):
    df = read_counts_matrix(write_matrix("gene;s1;s2\nA;1;2\nB;3;4\n"))
    assert df.shape == (2, 3)
    assert df["Sample_1"].tolist() == [1.0, 3.0]


def test_space_separated_matrix(write_matrix):
    df = read_counts_matrix(write_matrix("gene s1\nA   3\nB   5\n"))
    assert df["Sample_1"].tolist() == [3.0, 5.0]


def test_comment_lines_are_skipped_not_read_as_header(write_matrix):
    """A featureCounts banner must not become the header row."""
    df = read_counts_matrix(write_matrix(
        "# featureCounts 1.6.3\n# another banner\ngene\ts1\ts2\nA\t1\t2\n"))
    assert df["gene"].tolist() == ["A"]
    assert df["Sample_1"].tolist() == [1.0]


def test_zero_counts_matrix_is_still_a_matrix(write_matrix):
    """All-zero columns are real data (a gene nobody sequenced), not a refusal."""
    df = read_counts_matrix(write_matrix("gene\ts1\ts2\nA\t0\t0\nB\t0\t0\n"))
    assert df[["Sample_1", "Sample_2"]].values.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_returned_frame_is_ready_for_the_normalisers(write_matrix):
    """Whatever the input separator, the numeric columns are the samples, so
    cpm_normalization must find exactly those columns and scale them."""
    from biosuite.core.expression import cpm_normalization
    df = read_counts_matrix(write_matrix("gene\ts1\ts2\nA\t1\t3\nB\t3\t9\n"))
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    assert numeric == ["Sample_1", "Sample_2"]
    cpm = cpm_normalization(df)
    assert cpm["Sample_1"].sum() == pytest.approx(1e6)
    assert cpm["Sample_2"].sum() == pytest.approx(1e6)


# ── the R row-names headers ──────────────────────────────────────────────────

def test_header_with_leading_separator_and_matching_data(write_matrix):
    """write.table(row.names = TRUE) writes ",g1,g2" and the data rows carry the
    gene id in that first field; pandas already aligns this one."""
    df = read_counts_matrix(write_matrix(",g1,g2\nA,10,20\nB,30,40\n"))
    assert list(df.columns) == ["gene", "Sample_1", "Sample_2"]
    assert df["gene"].tolist() == ["A", "B"]
    assert df["Sample_1"].tolist() == [10.0, 30.0]


def test_header_with_leading_separator_but_no_row_names_in_data(write_matrix):
    """write.table(col.names = NA, row.names = FALSE) emits a header field the
    data rows never fill, which used to shift every name one column right and
    append an all-NaN sample: gene length would be read as expression."""
    df = read_counts_matrix(write_matrix("\tgene\ts1\ts2\nA\t1\t2\nB\t3\t4\n"))
    assert list(df.columns) == ["gene", "Sample_1", "Sample_2"]
    assert df["gene"].tolist() == ["A", "B"]
    assert df["Sample_1"].tolist() == [1.0, 3.0]
    assert df["Sample_2"].tolist() == [2.0, 4.0]


def test_shifted_header_with_metadata_column_is_realigned(write_matrix):
    """The same defect with a Length column: the recovered frame keeps the four
    declared columns and no phantom fifth one."""
    df = read_counts_matrix(write_matrix(
        "# fc\n\tgene\tLen\ts1\ts2\nA\t100\t1\t2\nB\t200\t3\t4\n"))
    assert list(df.columns) == ["gene", "Sample_1", "Sample_2", "Sample_3"]
    assert df["gene"].tolist() == ["A", "B"]
    assert df["Sample_2"].tolist() == [1.0, 3.0]   # s1, not shifted onto Len
    assert df["Sample_3"].tolist() == [2.0, 4.0]   # s2
    assert not df.isna().any().any(), "the phantom all-NaN sample must be gone"


def test_a_named_last_column_may_stay_all_nan(write_matrix):
    """Only a *header-alignment* artefact is dropped.  A sample the user named
    but left empty is their data, and renaming or deleting it here would hide a
    failed library."""
    df = read_counts_matrix(write_matrix("gene\ts1\ts2\nA\t1\t\nB\t3\t\n"))
    assert list(df.columns) == ["gene", "Sample_1", "Sample_2"]
    assert df["Sample_2"].isna().all()
    assert df["Sample_1"].tolist() == [1.0, 3.0]


# ── refusals: files that are not count matrices ──────────────────────────────

@pytest.mark.parametrize("text, why", [
    ("gene\nA\nB\n", "a bare gene list, sniffed into per-character columns"),
    ("counts\n5\n7\n", "one numeric column and no sample column"),
    ("gene,\nA,\nB,\n", "gene list wearing a trailing separator"),
    ("gene\t\nA\t\nB\t\n", "gene list with a trailing tab"),
    ('"gene"\n"A"\n"B"\n', "quoted single column, no delimiter anywhere"),
    ("gene\n", "a header and no data rows"),
])
def test_a_file_without_count_columns_is_refused(write_matrix, text, why):
    """The docstring promises None on error; before the alignment guard these
    returned a tidy-looking frame full of NaN instead."""
    assert read_counts_matrix(write_matrix(text)) is None, why


def test_only_comment_lines_is_refused(write_matrix):
    assert read_counts_matrix(write_matrix("#a\n#b\n")) is None


def test_missing_file_is_refused(tmp_path):
    assert read_counts_matrix(str(tmp_path / "absent.csv")) is None
