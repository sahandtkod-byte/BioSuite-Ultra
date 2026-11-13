"""The BigWig readers: a region request, a missing decoder, and the bin table.

bigWigReader is an optional package, so this module's honest answer when it is
absent matters more than usual — the old fallback returned an empty coverage array
and the CLI and GUI both printed that as "No BigWig data available." for a file
that was full of data.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core import file_formats as ff


BIGWIG_HEAD = bytes([0xC7, 0x4B, 0x6C, 0x27])


class StubBigWig:
    """bigWigReader-style handle: chr1 has entries, chr2 has none."""

    def __init__(self, path):
        self.path = path

    def getChroms(self):
        return ["chr1", "chr2"]

    def getChromSize(self, chrom):
        return 10 if chrom == "chr1" else 0

    def getEntries(self, chrom, start, end):
        if chrom == "chr2":
            return []
        return [(start, end, 1.5), (start + 1, end, 2.5)]


def _bigwig(tmp_path, name="signal.bw", head=BIGWIG_HEAD):
    path = tmp_path / name
    path.write_bytes(head)
    return str(path)


@pytest.fixture()
def no_decoder(monkeypatch):
    monkeypatch.setitem(sys.modules, "bigWigReader", None)


@pytest.fixture()
def stub_decoder(monkeypatch):
    module = types.ModuleType("bigWigReader")
    module.open = lambda path: StubBigWig(path)
    monkeypatch.setitem(sys.modules, "bigWigReader", module)
    return module


def test_a_region_read_returns_the_values_of_that_interval(stub_decoder, tmp_path):
    out = ff.read_bigwig(_bigwig(tmp_path), chrom="chr1", start=0, end=5)
    assert (out["chrom"], out["start"], out["end"]) == ("chr1", 0, 5)
    np.testing.assert_array_equal(out["values"], np.array([1.5, 2.5]))


def test_an_interval_with_no_entries_returns_an_empty_float_array(stub_decoder, tmp_path):
    out = ff.read_bigwig(_bigwig(tmp_path), chrom="chr2", start=0, end=5)
    assert out["values"].size == 0 and out["values"].dtype == np.float64


def test_a_read_without_a_region_returns_every_contig_the_header_lists(stub_decoder, tmp_path):
    out = ff.read_bigwig(_bigwig(tmp_path))
    assert set(out) == {"chroms"}
    np.testing.assert_array_equal(out["chroms"]["chr1"], np.array([1.5, 2.5]))
    assert out["chroms"]["chr2"].size == 0


@pytest.mark.parametrize("kwargs,missing", [
    ({"chrom": "chr1", "start": 100}, "only chrom, start were given"),
    ({"chrom": "chr1"}, "only chrom were given"),
    ({"start": 1, "end": 10}, "only start, end were given"),
])
def test_half_a_region_is_refused_instead_of_reading_the_whole_genome(
        stub_decoder, tmp_path, kwargs, missing):
    """A name without an interval used to fall through to the whole-file branch.

    The caller asked for one chromosome and got a dict keyed 'chroms' holding every
    contig in the file, which is the opposite of the argument it passed.
    """
    path = _bigwig(tmp_path)
    with pytest.raises(ValueError) as exc:
        ff.read_bigwig(path, **kwargs)
    assert "a BigWig region needs chrom, start and end together" in str(exc.value)
    assert missing in str(exc.value)


def test_a_real_bigwig_without_the_decoder_says_the_decoder_is_missing(
        no_decoder, tmp_path):
    """Not 'no data': the file is fine, the reader is what is absent.

    The fallback this replaces returned {'values': array([]), 'note': 'Basic BigWig
    parsing'}, and both front ends printed that as an empty experiment.
    """
    out = ff.read_bigwig(_bigwig(tmp_path))
    assert out["error"].endswith(
        "is a BigWig file, but BioSuite cannot decode it without the bigWigReader "
        "package. Run: pip install bigWigReader")
    assert "values" not in out and "chroms" not in out


def test_a_file_that_is_not_a_bigwig_is_named_with_the_bytes_it_found(no_decoder, tmp_path):
    path = _bigwig(tmp_path, name="bad.bw", head=b"BEDGraph")
    out = ff.read_bigwig(path)
    assert "does not start with the BigWig magic number c74b6c27 (found 42454447)" in out["error"]
    assert out["error"].endswith("so it is not a BigWig file")


def test_an_empty_file_reports_that_it_held_nothing_at_all(no_decoder, tmp_path):
    out = ff.read_bigwig(_bigwig(tmp_path, name="empty.bw", head=b""))
    assert "(found nothing)" in out["error"]


def test_a_missing_file_is_an_error_dict_naming_the_path(no_decoder, tmp_path):
    missing = str(tmp_path / "gone.bw")
    out = ff.read_bigwig(missing)
    assert out["error"].startswith(f"Could not open BigWig file {missing}")
    assert "No such file or directory" in out["error"]


def test_a_decoder_that_cannot_read_the_file_is_reported_not_raised(tmp_path, monkeypatch):
    module = types.ModuleType("bigWigReader")

    def explode(path):
        raise OSError("truncated header")

    module.open = explode
    monkeypatch.setitem(sys.modules, "bigWigReader", module)
    out = ff.read_bigwig(_bigwig(tmp_path))
    assert out == {"error": f"Could not read BigWig file {str(tmp_path / 'signal.bw')}: "
                            "truncated header"}


def test_the_summary_bins_coverage_and_keeps_the_short_tail_bin(monkeypatch):
    monkeypatch.setattr(ff, "read_bigwig", lambda path, **kw: {
        "chroms": {"chr1": np.arange(1, 8, dtype=float), "chr2": np.array([])}})
    df = ff.bigwig_summary("x.bw", bin_size=3)
    assert list(df.columns) == ["chrom", "start", "end", "mean", "max"]
    assert [(r["start"], r["end"], r["mean"], r["max"]) for r in df.to_dict("records")] == [
        (0, 3, 2.0, 3.0), (3, 6, 5.0, 6.0), (6, 7, 7.0, 7.0)]


def test_the_summary_of_a_contig_with_no_coverage_adds_no_rows(monkeypatch):
    monkeypatch.setattr(ff, "read_bigwig",
                        lambda path, **kw: {"chroms": {"chr1": np.array([])}})
    df = ff.bigwig_summary("x.bw")
    assert df.empty and list(df.columns) == ["chrom", "start", "end", "mean", "max"]


@pytest.mark.parametrize("error", [
    "is a BigWig file, but BioSuite cannot decode it",
    "Could not open BigWig file",
])
def test_an_unreadable_bigwig_becomes_an_empty_frame_and_the_reason_is_logged(
        monkeypatch, caplog, error):
    monkeypatch.setattr(ff, "read_bigwig", lambda path, **kw: {"error": error})
    with caplog.at_level("ERROR", logger="biosuite.core.file_formats"):
        df = ff.bigwig_summary("x.bw")
    assert df.empty and list(df.columns) == ["chrom", "start", "end", "mean", "max"]
    assert f"No BigWig summary of x.bw: {error}" in caplog.text, caplog.text


def test_a_reader_that_returned_no_contigs_says_that_instead_of_returning_nothing(
        monkeypatch, caplog):
    monkeypatch.setattr(ff, "read_bigwig",
                        lambda path, **kw: {"chrom": "chr1", "values": np.ones(3)})
    with caplog.at_level("ERROR", logger="biosuite.core.file_formats"):
        assert ff.bigwig_summary("x.bw").empty
    assert "the reader returned no per-contig coverage" in caplog.text, caplog.text


@pytest.mark.parametrize("bin_size", [0, -100, 0.5, None, True])
def test_a_bin_size_that_is_not_a_whole_positive_number_is_refused(bin_size):
    """Zero divided by zero, and negative bins labelled with a negative end.

    A bin_size of -100 over 500 values used to answer with one row reading
    start=0, end=-100 whose mean covered the first 400 positions — a statistic
    nothing in the file describes.
    """
    with pytest.raises(ValueError) as exc:
        ff.bigwig_summary("x.bw", bin_size=bin_size)
    assert f"bin_size must be a whole number of at least 1 bp, got {bin_size!r}" in str(exc.value)


def test_the_summary_text_announces_the_bins_it_holds_and_shows_the_first_twenty():
    df = pd.DataFrame([{"chrom": "chr1", "start": i * 10, "end": i * 10 + 10,
                        "mean": 1.25 + i, "max": 2.0 + i} for i in range(25)])
    lines = ff.format_bigwig_summary(df).splitlines()
    assert lines[0] == "BigWig Summary: 25 bins"
    assert lines[1] == "  chr1:0-10 mean=1.25 max=2.00"
    assert lines[20] == "  chr1:190-200 mean=20.25 max=21.00"
    assert lines[21] == "... and 5 more bin(s)."
    assert len(lines) == 22


def test_the_summary_text_says_nothing_was_truncated_when_every_bin_fit():
    df = pd.DataFrame([{"chrom": "chr1", "start": 0, "end": 10, "mean": 1.0, "max": 2.0}] * 3)
    text = ff.format_bigwig_summary(df)
    assert "more bin" not in text
    assert text.count("chr1:") == 3


def test_the_summary_text_of_an_empty_frame_says_there_is_no_data():
    assert ff.format_bigwig_summary(pd.DataFrame()) == "No BigWig data available."
    assert ff.format_bigwig_summary(
        pd.DataFrame(columns=["chrom", "start", "end", "mean", "max"])
    ) == "No BigWig data available."
