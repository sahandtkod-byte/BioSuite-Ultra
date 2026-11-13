"""The pysam-backed alignment readers (BAM and CRAM).

pysam is optional, so these two readers have exactly three outcomes: a real read,
an engine that is not installed, and an engine that tried and failed — and the
summary each returns is the only thing a user sees of that difference.
"""
import sys
import types

import pytest

from biosuite.core import file_formats as ff


class StubRead:
    """One pysam AlignedSegment, canned."""

    def __init__(self, index):
        self.query_name = f"read{index}"
        self.reference_start = index * 10
        self.reference_end = index * 10 + 50
        self.mapping_quality = 60
        self.query_alignment_length = 50
        self.is_unmapped = False
        self.is_reverse = bool(index % 2)


class StubAlignmentFile:
    opened = []            # [(path, mode, kwargs)] as pysam would receive them
    fetched = []           # region arguments, in call order

    def __init__(self, path, mode, **kwargs):
        StubAlignmentFile.opened.append((path, mode, kwargs))
        self.header = {"SQ": [{"SN": "chr1", "LN": 1000}], "HD": {"VN": "1.6"}}
        self.mapped = 7
        self.unmapped = 3

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def fetch(self, region=None):
        StubAlignmentFile.fetched.append(region)
        return [StubRead(i) for i in range(2)]


@pytest.fixture()
def fake_pysam(monkeypatch):
    StubAlignmentFile.opened = []
    StubAlignmentFile.fetched = []
    module = types.ModuleType("pysam")
    module.AlignmentFile = StubAlignmentFile
    monkeypatch.setitem(sys.modules, "pysam", module)
    return module


@pytest.fixture()
def no_pysam(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", None)


def test_a_cram_is_opened_in_cram_mode_with_the_reference_it_was_given(fake_pysam):
    out = ff.read_cram("a.cram", reference="ref.fa", region="chr1:1-100")
    assert StubAlignmentFile.opened == [
        ("a.cram", "rc", {"format": "cram", "reference_filename": "ref.fa"})]
    assert out["format"] == "cram" and out["reference"] == "ref.fa"
    assert StubAlignmentFile.fetched == ["chr1:1-100"]
    assert out["alignments"] == [
        {"query_name": "read0", "reference_start": 0, "reference_end": 50,
         "mapping_quality": 60, "query_alignment_length": 50,
         "is_unmapped": False, "is_reverse": False},
        {"query_name": "read1", "reference_start": 10, "reference_end": 60,
         "mapping_quality": 60, "query_alignment_length": 50,
         "is_unmapped": False, "is_reverse": True}]


def test_a_bam_is_opened_in_binary_mode_with_no_cram_keywords(fake_pysam):
    out = ff.read_bam("a.bam", region="chr1")
    assert StubAlignmentFile.opened == [("a.bam", "rb", {})]
    assert out["format"] == "bam" and out["reference"] is None
    assert len(out["alignments"]) == 2


def test_the_reference_is_only_asked_for_when_one_was_given(fake_pysam):
    ff.read_cram("a.cram")
    assert StubAlignmentFile.opened == [("a.cram", "rc", {"format": "cram"})]


def test_a_cram_without_a_reference_still_asks_pysam_to_find_one(fake_pysam):
    ff.read_cram("a.cram", region="chr1:1-10")
    assert "reference_filename" not in StubAlignmentFile.opened[0][2]


def test_the_counts_add_mapped_and_unmapped_into_a_total(fake_pysam):
    out = ff.read_cram("a.cram")
    assert (out["mapped_reads"], out["unmapped_reads"], out["total_reads"]) == (7, 3, 10)
    assert out["header"]["SQ"] == [{"SN": "chr1", "LN": 1000}]


def test_an_empty_alignment_list_without_a_region_says_which_region_was_asked(fake_pysam):
    """The list is empty because no interval was requested, not because there were
    no reads. Without the 'region' key the two are indistinguishable in the result.
    """
    out = ff.read_cram("a.cram")
    assert out["alignments"] == [] and out["region"] is None
    assert StubAlignmentFile.fetched == []


def test_the_region_that_produced_the_alignments_travels_with_them(fake_pysam):
    assert ff.read_bam("a.bam", region="chr2:1-50")["region"] == "chr2:1-50"


def test_a_pysam_failure_becomes_an_error_dict_naming_the_format(fake_pysam, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise ValueError("file is not BAM")

    monkeypatch.setattr(fake_pysam, "AlignmentFile", explode)
    with caplog.at_level("ERROR", logger="biosuite.core.file_formats"):
        out = ff.read_bam("a.bam")
    assert out == {"error": "file is not BAM", "format": "bam"}
    assert "Could not read the bam file a.bam: file is not BAM" in caplog.text, caplog.text


def test_without_pysam_the_bam_reader_names_the_package_it_needed(no_pysam):
    assert ff.read_bam("a.bam") == {
        "error": "pysam required for BAM reading. pip install pysam",
        "format": "bam", "note": "Install pysam for full BAM support"}


def test_without_pysam_the_cram_reader_names_the_package_it_needed(no_pysam):
    assert ff.read_cram("a.cram", reference="ref.fa") == {
        "error": "pysam required for CRAM reading. pip install pysam",
        "format": "cram", "note": "Install pysam for full CRAM support"}


def test_a_missing_pysam_is_not_reported_as_a_file_with_no_reads(no_pysam):
    """The docstring used to promise 'metadata only' here; there is no metadata.

    An alignment file that was never opened has to arrive as an error, because a
    dict of empty counts would read like an experiment that produced nothing.
    """
    out = ff.read_bam("a.bam")
    assert "mapped_reads" not in out and "alignments" not in out
    assert "pip install pysam" in out["error"]
