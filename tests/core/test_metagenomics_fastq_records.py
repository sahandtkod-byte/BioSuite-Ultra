"""The FASTQ reader behind metagenomics' built-in classifier.

``biosuite.core.metagenomics._read_fastq_records`` used to take any four lines in
order, so an aligned FASTA was parsed into read ids like '>s1' while half of its
records were swallowed as quality lines, and a truncated record in the middle of a
file dropped everything after it without a word.
"""
import logging

import pytest

from biosuite.core import metagenomics as mg


@pytest.fixture
def fastq(tmp_path):
    def _write(text, name="reads.fastq"):
        path = tmp_path / name
        path.write_text(text, newline="")
        return str(path)
    return _write


def test_four_lines_per_record_yield_the_id_and_the_sequence(fastq):
    path = fastq("@r1\nATCGATCGATCGATCG\n+\nIIII\n@r2\nGCTAGCTAGCTAGCTA\n+\nIIII\n")
    assert list(mg._read_fastq_records(path)) == [
        ("r1", "ATCGATCGATCGATCG"), ("r2", "GCTAGCTAGCTAGCTA")]


def test_the_read_id_is_the_first_token_of_the_header(fastq):
    path = fastq("@read42 some description here\nATCG\n+\nIIII\n")
    assert list(mg._read_fastq_records(path)) == [("read42", "ATCG")]


def test_a_fasta_file_is_named_as_the_wrong_format(fastq):
    """The most likely mistake at the CLI prompt is pointing at a FASTA; the old
    reader answered it with half the records and '>' prefixes as ids."""
    path = fastq(">s1\nACGTACGTAA\n>s2\nACGTACGTAC\n", name="aln.fasta")
    with pytest.raises(ValueError, match="is not a FASTQ file"):
        list(mg._read_fastq_records(path))
    with pytest.raises(ValueError, match=r"first line is '>s1'"):
        list(mg._read_fastq_records(path))


def test_a_header_that_loses_its_at_sign_stops_the_reader(fastq, caplog):
    path = fastq("@r1\nAAAA\n+\nIIII\nno_sign\nCCCC\n+\nIIII\n")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == [("r1", "AAAA")]
    assert "read header without its leading '@'" in caplog.text
    # The header line itself is already consumed, so only r2's three remaining
    # lines are reported as unread.
    assert "3 line(s) of the file were not classified" in caplog.text


def test_a_record_with_an_empty_sequence_stops_the_reader(fastq, caplog):
    path = fastq("@r1\nAAAA\n+\nIIII\n@r2\n\n+\nII\n@r3\nCCCC\n+\nIIII\n")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == [("r1", "AAAA")]
    assert "a record with an empty sequence line" in caplog.text
    # r3's four lines are what is left unread once r2 abandons the record loop.
    assert "4 line(s) of the file were not classified" in caplog.text


def test_a_quality_separator_that_is_not_a_plus_stops_the_reader(fastq, caplog):
    path = fastq("@r1\nAAAA\n!\nIIII\n@r2\nCCCC\n+\nIIII\n")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == []
    assert "quality separator line that reads '!'" in caplog.text


def test_a_short_final_record_is_a_truncated_file(fastq, caplog):
    """The file ends mid-record: what was read stays, the rest is reported."""
    path = fastq("@r1\nAAAA\n+\nIIII\n@r2\nCCCC\n")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == [("r1", "AAAA")]
    assert "stopped being read" in caplog.text


def test_an_empty_file_reads_no_records_and_says_nothing(fastq, caplog):
    path = fastq("")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == []
    assert caplog.text == ""


def test_a_complete_file_leaves_no_warning(fastq, caplog):
    path = fastq("@r1\nAAAA\n+\nIIII\n")
    with caplog.at_level(logging.WARNING):
        assert list(mg._read_fastq_records(path)) == [("r1", "AAAA")]
    assert caplog.text == ""


def test_crlf_line_endings_still_yield_clean_sequences(fastq):
    path = fastq("@r1\r\nATCGATCGATCGATCG\r\n+\r\nIIII\r\n")
    assert list(mg._read_fastq_records(path)) == [("r1", "ATCGATCGATCGATCG")]


def test_the_quality_string_is_not_used_as_a_sequence(fastq):
    """A reader that lost its frame would classify the quality line; the
    separator check is what makes that impossible."""
    path = fastq("@r1\nATCGATCGATCGATCG\n+\nIIII\n@r2\nIIII\n+\nATCGATCGATCGATCG\n")
    assert list(mg._read_fastq_records(path)) == [
        ("r1", "ATCGATCGATCGATCG"), ("r2", "IIII")]
