"""The three record readers in ``biosuite.core.sequence`` and what they do to bad input.

``read_fasta`` and ``read_fastq`` are pure Python and promise ``None`` on a file
they cannot read; ``read_genbank`` hands the job to Biopython and additionally
promises ``None`` when Biopython is not installed.  Each of those exits is a
different promise, and a FASTQ that loses records must say which ones it lost —
silently shortening a run's read count is how a sequencing analysis goes wrong
without ever raising.
"""
import importlib.util
import pathlib
import sys

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from biosuite.core import sequence as seq_mod

FASTQ_ONE_RECORD = "@read1\nACGT\n+\nIIII\n"


@pytest.fixture
def genbank_file(tmp_path):
    """A real GenBank file with one CDS, written by Biopython itself."""
    def write(name="demo.gb"):
        record = SeqRecord(Seq("ATGAAAGGGTTTAA"), id="T_001", name="demo",
                           description="a demo record")
        record.annotations["molecule_type"] = "DNA"
        record.features.append(SeqFeature(
            FeatureLocation(0, 12), type="CDS", qualifiers={"gene": ["ake"]}))
        path = tmp_path / name
        SeqIO.write(record, str(path), "genbank")
        return str(path)

    return write


@pytest.fixture(scope="module")
def without_biopython():
    """``sequence`` executed a second time with Biopython hidden from the imports."""
    path = pathlib.Path(seq_mod.__file__)
    saved = sys.modules.get("Bio")
    sys.modules["Bio"] = None
    try:
        spec = importlib.util.spec_from_file_location(
            "biosuite.core.sequence_without_biopython", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("Bio", None)
        else:
            sys.modules["Bio"] = saved
    return module


# ─── FASTA ───────────────────────────────────────────────────────────────────

def test_blank_lines_between_records_are_not_a_record_of_their_own(tmp_path):
    path = tmp_path / "two.fa"
    path.write_text(">first\nACGT\n\n>second\nGGGG\n\n", encoding="utf-8")

    assert seq_mod.read_fasta(str(path)) == [("first", "ACGT"), ("second", "GGGG")]


def test_a_file_that_lies_about_being_compressed_is_refused(caplog, tmp_path):
    path = tmp_path / "liar.fa.gz"
    path.write_bytes(b"not really gzip at all")

    with caplog.at_level("ERROR", logger="biosuite.core.sequence"):
        assert seq_mod.read_fasta(str(path)) is None

    assert "Error reading FASTA" in caplog.text, caplog.text


def test_a_missing_fasta_is_answered_with_nothing_and_no_message(tmp_path):
    assert seq_mod.read_fasta(str(tmp_path / "absent.fa")) is None


# ─── FASTQ ───────────────────────────────────────────────────────────────────

def test_a_record_whose_header_lost_its_at_sign_is_reported_and_dropped(tmp_path, caplog):
    path = tmp_path / "broken.fq"
    path.write_text("read1 no at sign\nACGT\n+\nIIII\n" + FASTQ_ONE_RECORD,
                    encoding="utf-8")

    with caplog.at_level("WARNING", logger="biosuite.core.sequence"):
        records = seq_mod.read_fastq(str(path))

    assert [r[0] for r in records] == ["read1"], "the well-formed record survived"
    assert "skipped 1 malformed record" in caplog.text, caplog.text
    assert "does not start with '@'" in caplog.text, caplog.text


def test_a_record_whose_separator_line_is_not_a_plus_is_refused_by_name(tmp_path, caplog):
    path = tmp_path / "badplus.fq"
    path.write_text(FASTQ_ONE_RECORD + "@read2\nACGT\nx\nIIII\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger="biosuite.core.sequence"):
        records = seq_mod.read_fastq(str(path))

    assert [r[0] for r in records] == ["read1"], records
    assert "record 2: separator line is 'x', expected '+'" in caplog.text, caplog.text


def test_a_record_whose_quality_is_a_different_length_is_refused_by_name(tmp_path, caplog):
    path = tmp_path / "badqual.fq"
    path.write_text(FASTQ_ONE_RECORD + "@read2\nACGT\n+\nIII\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger="biosuite.core.sequence"):
        records = seq_mod.read_fastq(str(path))

    assert [r[0] for r in records] == ["read1"], records
    assert "record 2: sequence length 4 != quality length 3" in caplog.text, caplog.text
    assert "skipped 1 malformed record" in caplog.text, caplog.text


def test_a_blank_line_where_a_header_belongs_is_not_counted_as_a_record(tmp_path, caplog):
    path = tmp_path / "spaced.fq"
    path.write_text(FASTQ_ONE_RECORD + "\n   \n" + FASTQ_ONE_RECORD.replace("read1", "read2"),
                    encoding="utf-8")

    with caplog.at_level("WARNING", logger="biosuite.core.sequence"):
        records = seq_mod.read_fastq(str(path))

    assert [r[0] for r in records] == ["read1", "read2"], records
    assert "skipped" not in caplog.text, \
        "an empty line is not a record, so nothing was reported as lost"


def test_a_file_that_lies_about_being_compressed_is_answered_with_none(caplog, tmp_path):
    path = tmp_path / "unreadable.fq.gz"
    path.write_bytes(b"definitely not gzip")

    with caplog.at_level("ERROR", logger="biosuite.core.sequence"):
        assert seq_mod.read_fastq(str(path)) is None

    assert "Error reading FASTQ" in caplog.text, caplog.text


# ─── GenBank ─────────────────────────────────────────────────────────────────

def test_a_genbank_record_keeps_its_identifier_sequence_and_features(genbank_file):
    records = seq_mod.read_genbank(genbank_file())

    assert len(records) == 1
    record_id, sequence, features = records[0]
    assert (record_id, sequence) == ("T_001", "ATGAAAGGGTTTAA")
    assert [f["type"] for f in features] == ["CDS"]
    assert features[0]["qualifiers"] == {"gene": ["ake"]}
    assert features[0]["location"] == "[0:12](+)"


def test_a_genbank_file_that_is_not_there_is_answered_with_none(tmp_path):
    assert seq_mod.read_genbank(str(tmp_path / "absent.gb")) is None


def test_a_path_that_is_not_a_file_is_reported_instead_of_passing_as_empty(tmp_path, caplog):
    directory = tmp_path / "not_a_file.gb"
    directory.mkdir()

    with caplog.at_level("ERROR", logger="biosuite.core.sequence"):
        assert seq_mod.read_genbank(str(directory)) is None

    assert "Error reading GenBank" in caplog.text, caplog.text


def test_genbank_reading_says_so_when_biopython_is_not_installed(without_biopython, caplog):
    assert without_biopython.HAS_BIO is False

    with caplog.at_level("WARNING", logger="biosuite.core.sequence"):
        assert without_biopython.read_genbank("anything.gb") is None

    assert "Biopython not installed" in caplog.text, caplog.text


def test_the_rest_of_the_module_still_works_without_biopython(without_biopython):
    assert without_biopython.gc_content("GGCC") == 100.0
    assert without_biopython.reverse_complement("ACGT") == "ACGT"


# ─── the two small guards ────────────────────────────────────────────────────

def test_a_sequence_of_nothing_but_gaps_has_no_gc_to_report():
    assert seq_mod.gc_content("--..*") == 0.0
    assert seq_mod.gc_content("") == 0.0


def test_an_empty_quality_string_reports_a_run_with_no_bases():
    assert seq_mod.quality_stats("") == {
        'mean': 0.0, 'min': 0, 'max': 0, 'positions': [], 'scores': []}
