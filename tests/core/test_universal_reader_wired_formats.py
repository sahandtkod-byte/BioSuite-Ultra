"""read_file's wiring for the four formats it used to detect but not read.

Before #51 a .sam/.bcf/.newick/.bw file was detected correctly and then
refused with "no reader for ... files".  Each now has a reader, and these
tests pin what a caller gets back for a real file, a missing decoder and a
missing file.
"""
import sys
import types

import pytest

from biosuite.core import file_formats as ff


BIGWIG_HEAD = bytes([0xC7, 0x4B, 0x6C, 0x27])

SAM_TEXT = (
    "@HD\tVN:1.6\n"
    "@SQ\tSN:chr1\tLN:1000\n"
    "read1\t0\tchr1\t100\t60\t10M\t*\t0\t0\tACGTACGTAC\tIIIIIIIIII\n"
)


def _write(tmp_path, name, content, binary=False):
    path = tmp_path / name
    if binary:
        path.write_bytes(content)
    else:
        path.write_text(content)
    return str(path)


@pytest.fixture()
def no_pysam(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", None)


@pytest.fixture()
def no_bigwig_decoder(monkeypatch):
    monkeypatch.setitem(sys.modules, "bigWigReader", None)


# ── SAM ──────────────────────────────────────────────────────────────────────

def test_a_sam_file_comes_back_as_records_carrying_the_qname(tmp_path):
    out = ff.read_file(_write(tmp_path, "a.sam", SAM_TEXT))
    assert out["format"] == "sam"
    assert out["records"][0]["QNAME"] == "read1"
    assert out["records"][0]["RNAME"] == "chr1"


def test_a_missing_sam_file_is_an_error_naming_the_path(tmp_path):
    missing = str(tmp_path / "gone.sam")
    out = ff.read_file(missing)
    assert out["format"] == "sam"
    assert f"{missing} was not read as SAM" in out["error"]


def test_a_sam_option_is_refused_by_name(tmp_path):
    path = _write(tmp_path, "a.sam", SAM_TEXT)
    out = ff.read_file(path, region="chr1:1-10")
    assert out["format"] == "sam"
    assert "The sam reader does not take region (it accepts: no options)" in out["error"]


# ── Newick ───────────────────────────────────────────────────────────────────

def test_a_newick_file_comes_back_as_a_tree_with_children(tmp_path):
    out = ff.read_file(_write(tmp_path, "a.nwk", "(A:0.1,B:0.2)C:0.3;\n"))
    assert out["format"] == "newick"
    assert [c.name for c in out["tree"].children] == ["A", "B"]


def test_a_malformed_newick_file_is_an_error_naming_the_tree_problem(tmp_path):
    out = ff.read_file(_write(tmp_path, "bad.nwk", "(A,B\n"))
    assert out["format"] == "newick"
    assert out["error"].startswith("Not a Newick tree:")


def test_an_empty_newick_file_is_refused_with_the_path(tmp_path):
    path = _write(tmp_path, "empty.nwk", "")
    out = ff.read_file(path)
    assert out["format"] == "newick"
    assert f"{path} is empty" in out["error"]


def test_a_missing_newick_file_is_an_error_naming_the_path(tmp_path):
    missing = str(tmp_path / "gone.nwk")
    out = ff.read_file(missing)
    assert out["format"] == "newick"
    assert out["error"].startswith(f"Could not read the Newick file {missing}")


# ── BCF ──────────────────────────────────────────────────────────────────────

def test_a_bcf_without_pysam_names_the_install_command(tmp_path, no_pysam):
    out = ff.read_file(_write(tmp_path, "a.bcf", ff._BCF_MAGIC + b"junk", binary=True))
    assert out["format"] == "bcf"
    assert "pip install pysam" in out["data"]["error"]


def test_a_bgzf_headed_bcf_is_recognised_not_rejected_on_its_magic(
        tmp_path, no_pysam):
    """A real BCF on disk is BGZF-compressed, so its head is a gzip header."""
    out = ff.read_file(_write(tmp_path, "a.bcf", b"\x1f\x8b\x08\x04junk", binary=True))
    assert out["format"] == "bcf"
    assert "pip install pysam" in out["data"]["error"]


def test_a_file_that_is_not_a_bcf_is_refused_on_its_magic(tmp_path):
    out = ff.read_file(_write(tmp_path, "bad.bcf", b"\x00\x00\x00\x00", binary=True))
    assert out["format"] == "bcf"
    assert "does not start with the BCF magic" in out["data"]["error"]


def test_an_empty_bcf_file_says_it_held_nothing(tmp_path):
    out = ff.read_file(_write(tmp_path, "empty.bcf", b"", binary=True))
    assert out["format"] == "bcf"
    assert "(found nothing)" in out["data"]["error"]


def test_a_missing_bcf_file_is_an_error_naming_the_path(tmp_path):
    missing = str(tmp_path / "gone.bcf")
    out = ff.read_file(missing)
    assert out["format"] == "bcf"
    assert out["data"]["error"].startswith(f"Could not open BCF file {missing}")


class _FakeVariant:
    contig = "chr1"
    pos = 100
    id = "rs1"
    ref = "A"
    alts = ("T",)


class _BareVariant:
    contig = "chr2"
    pos = 7
    id = None
    ref = "G"
    alts = None


def test_a_bcf_with_pysam_present_returns_variant_records(tmp_path, monkeypatch):
    module = types.ModuleType("pysam")

    class FakeVariantFile:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([_FakeVariant(), _BareVariant()])

    module.VariantFile = FakeVariantFile
    monkeypatch.setitem(sys.modules, "pysam", module)
    out = ff.read_file(_write(tmp_path, "a.bcf", ff._BCF_MAGIC + b"junk", binary=True))
    assert out["format"] == "bcf"
    assert out["data"]["records"] == [
        {"chrom": "chr1", "pos": 100, "id": "rs1", "ref": "A", "alts": ["T"]},
        {"chrom": "chr2", "pos": 7, "id": "", "ref": "G", "alts": []},
    ]


def test_a_bcf_the_decoder_cannot_open_is_reported(tmp_path, monkeypatch):
    module = types.ModuleType("pysam")

    class BrokenVariantFile:
        def __init__(self, path):
            raise OSError("truncated bgzf block")

    module.VariantFile = BrokenVariantFile
    monkeypatch.setitem(sys.modules, "pysam", module)
    out = ff.read_file(_write(tmp_path, "a.bcf", ff._BCF_MAGIC + b"junk", binary=True))
    assert out["format"] == "bcf"
    assert "Could not decode BCF file" in out["data"]["error"]
    assert "truncated bgzf block" in out["data"]["error"]


# ── BigWig ───────────────────────────────────────────────────────────────────

def test_a_bigwig_without_the_decoder_names_the_missing_package(
        tmp_path, no_bigwig_decoder):
    out = ff.read_file(_write(tmp_path, "a.bw", BIGWIG_HEAD + b"junk", binary=True))
    assert out["format"] == "bigwig"
    assert "bigWigReader" in out["data"]["error"]


def test_a_file_that_is_not_a_bigwig_is_refused_on_its_magic(
        tmp_path, no_bigwig_decoder):
    out = ff.read_file(_write(tmp_path, "bad.bw", b"\x00\x00\x00\x00", binary=True))
    assert out["format"] == "bigwig"
    assert "does not start with the BigWig magic number" in out["data"]["error"]


def test_half_a_bigwig_region_is_refused_through_read_file(tmp_path):
    out = ff.read_file(_write(tmp_path, "a.bw", BIGWIG_HEAD + b"junk", binary=True),
                       chrom="chr1")
    assert out["format"] == "bigwig"
    assert "a BigWig region needs chrom, start and end together" in out["error"]


def test_a_bigwig_success_travels_under_the_data_key(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "read_bigwig",
                        lambda path, **kw: {"chroms": {"chr1": []}})
    out = ff.read_file(_write(tmp_path, "a.bw", BIGWIG_HEAD, binary=True))
    assert out == {"format": "bigwig", "data": {"chroms": {"chr1": []}}}


def test_bigwig_region_options_are_forwarded_to_the_reader(tmp_path, monkeypatch):
    seen = {}

    def fake_bigwig(path, **kw):
        seen.update(kw)
        return {"chrom": "chr1", "start": 0, "end": 5, "values": []}

    monkeypatch.setattr(ff, "read_bigwig", fake_bigwig)
    ff.read_file(_write(tmp_path, "a.bw", BIGWIG_HEAD, binary=True),
                 chrom="chr1", start=0, end=5)
    assert seen == {"chrom": "chr1", "start": 0, "end": 5}
