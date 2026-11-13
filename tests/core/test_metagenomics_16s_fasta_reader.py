"""The FASTA reader `classify_16s_rna()` uses when it is given a path.

A header line of bare '>' used to raise IndexError from inside the parser, sequence
lines before the first header were dropped in silence, and a file that did not exist
was reported as a completed 16S run of zero sequences.
"""
import logging

import pytest

from biosuite.core import metagenomics as mg

ECOLI = mg.SILVA_16S_DB["Escherichia coli"]


@pytest.fixture
def fasta(tmp_path):
    def _write(text, name="reads.fa"):
        path = tmp_path / name
        path.write_text(text, newline="")
        return str(path)
    return _write


def test_a_multi_record_file_is_read_with_its_wrapped_lines(fasta):
    path = fasta(">seq1 some description\nTGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA\n"
                 ">seq2\nTGGAGGAAGGTGGGG\nACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA\n")
    assert mg._read_16s_fasta(path) == [
        ("seq1", ECOLI),
        ("seq2", "TGGAGGAAGGTGGGG" + ECOLI[15:]),
    ]


def test_only_the_first_token_of_the_header_becomes_the_name(fasta):
    path = fasta(">read_7 flags=0 description here\nACGTACGTACGTACGTACGTACGT\n")
    assert mg._read_16s_fasta(path)[0][0] == "read_7"


def test_blank_lines_are_ignored_not_turned_into_sequences(fasta):
    path = fasta(">a\n\nACGTACGTACGTACGTACGTACGT\n\n\n")
    assert mg._read_16s_fasta(path) == [("a", "ACGTACGTACGTACGTACGTACGT")]


def test_a_header_with_no_name_is_reported_with_its_line_number(fasta):
    path = fasta(">\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")
    with pytest.raises(ValueError, match="line 1: a FASTA header with no name"):
        mg._read_16s_fasta(path)


def test_the_offending_line_number_is_the_real_one(fasta):
    path = fasta(">a\nACGTACGTACGTACGTACGTACGTACGT\n>\nb\nACGT\n")
    with pytest.raises(ValueError, match="line 3"):
        mg._read_16s_fasta(path)


def test_sequence_lines_before_the_first_header_are_counted_not_guessed(fasta, caplog):
    """'TGGAG...' with no name above it belongs to no record; inventing a name for
    it would put an untraceable sequence into the abundance table."""
    path = fasta("TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA\n")
    with caplog.at_level(logging.WARNING):
        assert mg._read_16s_fasta(path) == []
    assert "1 sequence line(s) before the first FASTA header" in caplog.text


def test_a_file_that_is_all_headers_reads_no_records(fasta):
    path = fasta(">a\n>b\n")
    assert mg._read_16s_fasta(path) == [("a", ""), ("b", "")]


def test_the_last_record_is_not_dropped(fasta):
    path = fasta(">a\nACGTACGTACGTACGTACGTACGTACGT\n>b\nTTTTTTTTTTTTTTTTTTTTTTTT\n")
    assert [name for name, _ in mg._read_16s_fasta(path)] == ["a", "b"]


# ── the public entry point on a path ────────────────────────────────────────

def test_a_fasta_path_is_classified_record_by_record(fasta):
    path = fasta(f">q1\n{ECOLI}\n>q2\n{'N' * len(ECOLI)}\n")
    res = mg.classify_16s_rna(path)
    assert [c["name"] for c in res.classifications] == ["q1", "q2"]
    assert [c["taxonomy"] for c in res.classifications] == ["Escherichia coli",
                                                            "Unclassified"]
    assert res.message == "Classified 2 16S sequences"


def test_a_missing_file_is_an_error_the_report_shows(fasta, tmp_path):
    missing = str(tmp_path / "absent.fa")
    res = mg.classify_16s_rna(missing)
    assert res.classifications == []
    assert res.message == f"File not found: {missing}"
    assert "File not found" in mg.format_16s_report(res)


def test_a_directory_is_not_quietly_a_zero_sequence_run(tmp_path):
    """The parser used to catch only FileNotFoundError, so opening a directory
    raised PermissionError through the middle of the CLI."""
    with pytest.raises((PermissionError, IsADirectoryError, OSError)):
        mg.classify_16s_rna(str(tmp_path))


def test_a_file_without_any_header_yields_no_sequences(fasta):
    path = fasta(f"{ECOLI}\n")
    res = mg.classify_16s_rna(path)
    assert res.classifications == []
    assert res.message == "Classified 0 16S sequences"
