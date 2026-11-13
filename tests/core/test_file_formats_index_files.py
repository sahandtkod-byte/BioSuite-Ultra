"""The .bai and .tbi index readers: what a file has to prove about itself.

These two report whether an index is usable, so the answer has to come from the
bytes of the index and its data file — not from which packages happen to be
importable on the machine that asked.
"""
import sys
import types

import pytest

from biosuite.core import file_formats as ff


BAM_MAGIC = bytes([0x42, 0x41, 0x4D, 0x01])       # b"BAM\x01", htslib's index magic
TABIX_MAGIC = bytes([0x54, 0x42, 0x49, 0x01])     # b"TBI\x01"


class StubTabix:
    opened = []
    closed = []

    def __init__(self, path):
        StubTabix.opened.append(path)
        self.contigs = ["chr1", "chr2"]

    def close(self):
        StubTabix.closed.append(self)


@pytest.fixture()
def fake_pysam(monkeypatch):
    StubTabix.opened = []
    StubTabix.closed = []
    module = types.ModuleType("pysam")
    module.TabixFile = StubTabix
    module.AlignmentFile = object
    monkeypatch.setitem(sys.modules, "pysam", module)
    return module


@pytest.fixture()
def no_pysam(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", None)


def _index(tmp_path, name, head):
    path = tmp_path / name
    path.write_bytes(head)
    return str(path)


def test_a_missing_index_file_is_reported_by_path(tmp_path):
    for reader, name in ((ff.read_bam_index, "x.bam.bai"), (ff.read_vcf_index, "x.vcf.tbi")):
        missing = str(tmp_path / name)
        assert reader(missing) == {"error": f"Index file not found: {missing}"}


def test_a_bam_index_is_valid_because_of_its_own_magic(fake_pysam, tmp_path):
    path = _index(tmp_path, "x.bam.bai", BAM_MAGIC + b"\x02\x03")
    out = ff.read_bam_index(path)
    assert out["valid"] is True
    assert out["format"] == "bai" and out["path"] == path
    assert out["file_size"] == 6
    assert out["note"] == "BAM index recognised. Use pysam for detailed queries."


def test_an_index_of_four_unrelated_bytes_is_not_a_bam_index(fake_pysam, tmp_path):
    """'valid' used to mean the import of pysam worked.

    A text file renamed .bai, and a truncated index, both came back valid, and the
    note said the index had loaded successfully when nothing had been read from it.
    """
    out = ff.read_bam_index(_index(tmp_path, "notes.bai", b"help"))
    assert out["valid"] is False
    assert "expected the magic b'BAM\\x01' at the head of the file, found b'help'" in out["note"]


def test_an_index_shorter_than_its_magic_is_reported_as_invalid(no_pysam, tmp_path):
    out = ff.read_bam_index(_index(tmp_path, "tiny.bai", b"BA"))
    assert out["valid"] is False and out["file_size"] == 2
    assert "found b'BA'" in out["note"]


def test_a_valid_bam_index_without_pysam_says_what_is_unavailable(no_pysam, tmp_path):
    out = ff.read_bam_index(_index(tmp_path, "x.bam.bai", BAM_MAGIC))
    assert out["valid"] is True
    assert out["note"] == "BAM index recognised. Install pysam to query it in detail."


def test_a_tabix_index_lists_the_contigs_of_the_data_file_beside_it(fake_pysam, tmp_path):
    tbi = _index(tmp_path, "x.vcf.gz.tbi", TABIX_MAGIC)
    (tmp_path / "x.vcf.gz").write_bytes(b"\x1f\x8b stuff")
    out = ff.read_vcf_index(tbi)
    assert out["valid"] is True
    assert out["chromosomes"] == ["chr1", "chr2"] and out["n_chromosomes"] == 2
    assert StubTabix.opened == [str(tmp_path / "x.vcf.gz")]
    assert len(StubTabix.closed) == 1, "the handle has to be closed"


def test_the_data_file_is_named_by_cutting_the_suffix_off_the_index_path(tmp_path):
    """A replace() over the whole path rewrote any '.tbi' it found, directory included.

    This is the case that used to produce a data path that could not exist:
    /run/tbi_dir/sample.vcf.gz.tbi became /run/vcf.gz_dir/sample.vcf.gz.vcf.gz.
    """
    assert ff._tabix_data_file("/run/x.vcf.gz.tbi") == "/run/x.vcf.gz"
    assert ff._tabix_data_file("/run/tbi_dir/sample.vcf.gz.tbi") == "/run/tbi_dir/sample.vcf.gz"
    assert ff._tabix_data_file("/run/sample.vcf.TBI") == "/run/sample.vcf"
    assert ff._tabix_data_file("/run/indexed.data") is None


def test_a_tabix_index_with_a_directory_named_like_the_suffix_still_reads(tmp_path, fake_pysam):
    folder = tmp_path / "tbi_backup"
    folder.mkdir()
    tbi = folder / "s.vcf.gz.tbi"
    tbi.write_bytes(TABIX_MAGIC)
    (folder / "s.vcf.gz").write_bytes(b"\x1f\x8b")
    out = ff.read_vcf_index(str(tbi))
    assert out["chromosomes"] == ["chr1", "chr2"]
    assert StubTabix.opened == [str(folder / "s.vcf.gz")]


def test_a_valid_tabix_index_whose_vcf_is_absent_says_so(no_pysam, tmp_path):
    out = ff.read_vcf_index(_index(tmp_path, "orphan.vcf.gz.tbi", TABIX_MAGIC))
    assert out["valid"] is True and "chromosomes" not in out
    assert "the data file" in out["note"] and "is not there, so no contigs could be listed" in out["note"]


def test_an_index_path_without_the_tbi_suffix_cannot_name_its_data_file(no_pysam, tmp_path):
    out = ff.read_vcf_index(_index(tmp_path, "no_suffix", TABIX_MAGIC))
    assert out["valid"] is True
    assert "does not end in '.tbi'" in out["note"]


def test_a_file_that_is_not_a_tabix_index_is_named_with_the_bytes_it_found(no_pysam, tmp_path):
    out = ff.read_vcf_index(_index(tmp_path, "junk.vcf.gz.tbi", b"junk"))
    assert out["valid"] is False
    assert "expected the magic b'TBI\\x01' at the head of the file, found b'junk'" in out["note"]


def test_a_valid_tabix_index_without_pysam_offers_no_contigs(no_pysam, tmp_path):
    tbi = _index(tmp_path, "x.vcf.gz.tbi", TABIX_MAGIC)
    (tmp_path / "x.vcf.gz").write_bytes(b"\x1f\x8b")
    out = ff.read_vcf_index(tbi)
    assert out["valid"] is True and "chromosomes" not in out
    assert out["note"] == "Index recognised. Install pysam to list its contigs."


def test_a_pysam_query_failure_is_reported_beside_the_valid_index(tmp_path, monkeypatch):
    module = types.ModuleType("pysam")

    def explode(path):
        raise ValueError("x.vcf.gz is not bgzip-compressed")

    module.TabixFile = explode
    monkeypatch.setitem(sys.modules, "pysam", module)
    tbi = _index(tmp_path, "x.vcf.gz.tbi", TABIX_MAGIC)
    (tmp_path / "x.vcf.gz").write_bytes(b"plain text vcf")
    out = ff.read_vcf_index(tbi)
    assert out["valid"] is True
    assert "pysam could not query" in out["note"] and "bgzip-compressed" in out["note"]


@pytest.mark.parametrize("name,reader,label,fmt", [
    ("locked.bai", ff.read_bam_index, "BAM index", "bai"),
    ("locked.vcf.gz.tbi", ff.read_vcf_index, "tabix index", "tbi"),
])
def test_an_unreadable_index_file_is_an_error_naming_the_path(tmp_path, monkeypatch,
                                                              name, reader, label, fmt):
    path = _index(tmp_path, name, BAM_MAGIC)

    def refuse(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", refuse)
    assert reader(path) == {"format": fmt, "path": path,
                            "error": f"Could not read the {label} {path}"}
