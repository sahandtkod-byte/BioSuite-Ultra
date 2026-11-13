"""`InputValidator.validate_fastq` — the four-line walk and the trimmer's agreement.

The trimmer in this package (`biosuite/core/trimming.py`) reads a `+read1` separator
line and a blank line between records, because both are legal FASTQ; this validator
used to refuse them, which meant a user who checked a file before trimming it was
told the file was broken and then, two screens later, got a clean trim report from
the same bytes.  The two now agree, and each disagreement that is still an error is
reported by the line the user sees in the file.
"""
import pytest

from biosuite.core import trimming as tr
from biosuite.core.validators import InputValidator


@pytest.fixture
def validator():
    return InputValidator()


# ── what a good file looks like ─────────────────────────────────────────────

def test_a_clean_record_and_a_clean_two_record_file_pass(validator):
    assert validator.validate_fastq("@r1\nACGT\n+\nIIII\n") == []
    assert validator.validate_fastq("@r1\nACGT\n+\nIIII\n@r2\nAC\n+\n!~\n") == []


def test_the_printable_phred_range_is_accepted_end_to_end(validator):
    assert validator.validate_fastq("@r1\nACGT\n+\n!~I6\n") == []


def test_the_repeated_header_separator_the_format_allows_is_not_an_error(validator):
    """Measured on the old code: `@r1/ACGT/+r1/IIII` — legal FASTQ, and a file
    `trimming.trim_fastq` reads without a word — came back as
    "Separator line must be '+'"."""
    assert validator.validate_fastq("@r1\nACGT\n+r1\nIIII\n") == []
    assert validator.validate_fastq("@r1 first read\nACGT\n+r1\nIIII\n") == []


def test_blank_lines_between_records_are_not_fields(validator):
    """The same shape `trimming` skips: a blank line where a header belongs."""
    assert validator.validate_fastq("@r1\nACGT\n+\nIIII\n\n@r2\nACGT\n+\nIIII\n") == []
    assert validator.validate_fastq("\n\n@r1\nACGT\n+\nIIII\n\n\n") == []


# ── the errors, numbered by the file ────────────────────────────────────────

@pytest.mark.parametrize("content", ["", "  \n  ", None])
def test_content_that_holds_nothing_is_not_a_file(validator, content):
    assert validator.validate_fastq(content) == ["FASTQ content is empty"]


def test_a_header_without_its_at_sign_is_named_with_its_line(validator):
    assert validator.validate_fastq("r1\nACGT\n+\nIIII\n") == [
        "Record 1 (line 1): Header must start with '@'"
    ]


def test_a_separator_that_is_neither_plus_nor_the_header_is_shown(validator):
    assert validator.validate_fastq("@r1\nACGT\n-\nIIII\n") == [
        "Record 1 (line 3): Separator line must be '+' or a repeat of the header, got '-'"
    ]
    assert validator.validate_fastq("@r1\nACGT\n+r2\nIIII\n") == [
        "Record 1 (line 3): Separator line must be '+' or a repeat of the header, got '+r2'"
    ]


def test_a_quality_line_of_the_wrong_length_reports_both_lengths(validator):
    assert validator.validate_fastq("@r1\nACGT\n+\nIII\n") == [
        "Record 1 (line 2): Sequence length (4) != quality length (3)"
    ]
    assert validator.validate_fastq("@r1\nACGT\n+\n\n@r2\nACGT\n+\nIIII\n") == [
        "Record 1 (line 2): Sequence length (4) != quality length (0)"
    ]
    assert validator.validate_fastq("@r1\n\n+\nIIII\n") == [
        "Record 1 (line 2): Sequence length (0) != quality length (4)"
    ]


@pytest.mark.parametrize("quality, shown", [
    pytest.param("\x01\x01\x01\x01", "'\\x01'", id="control-below"),
    pytest.param("\x7f\x7f\x7f\x7f", "'\\x7f'", id="del-above"),
    pytest.param("F IF", "' '", id="space"),
    pytest.param("\x1f\x1f\x1f\x1f", "'\\x1f'", id="unit-separator"),
])
def test_quality_characters_outside_phred_33_are_shown_by_character(validator, quality,
                                                                    shown):
    assert validator.validate_fastq(f"@r1\nACGT\n+\n{quality}\n") == [
        f"Record 1 (line 4): quality characters outside Phred+33: {shown}"
    ]


def test_a_file_that_is_not_a_whole_number_of_records_says_which_lines(validator):
    """Measured on the old code: three lines were reported as "3 extra line(s)",
    which reads as three lines too many when the record is one line short."""
    assert validator.validate_fastq("@r1\nACGT\n+\n") == [
        "FASTQ must have 4 lines per record; lines 1-3 are a truncated final record "
        "(3 of 4 lines)"
    ]
    assert validator.validate_fastq("@r1\nACGT\n+\nIIII\nX") == [
        "FASTQ must have 4 lines per record; lines 5-5 are a truncated final record "
        "(1 of 4 lines)"
    ]


def test_the_line_numbers_count_the_file_including_its_leading_blanks(validator):
    """Measured on the old code: the content was `strip()`ed before splitting, so
    the same problem was reported two lines earlier than the file shows it."""
    assert validator.validate_fastq("\n\n@r1\nACGT\n+\nIII\n") == [
        "Record 1 (line 4): Sequence length (4) != quality length (3)"
    ]


def test_every_record_after_the_first_is_checked_too(validator):
    assert validator.validate_fastq("x\nACG\n-\nFF\ny\nACGT\n+\nIII\n") == [
        "Record 1 (line 1): Header must start with '@'",
        "Record 1 (line 3): Separator line must be '+' or a repeat of the header, got '-'",
        "Record 1 (line 2): Sequence length (3) != quality length (2)",
        "Record 2 (line 5): Header must start with '@'",
        "Record 2 (line 6): Sequence length (4) != quality length (3)",
    ]


# ── the agreement with the engine that reads the file ───────────────────────

@pytest.mark.parametrize("content, refused", [
    pytest.param("@r1\nACGT\n+r1\nIIII\n", False, id="repeated-header"),
    pytest.param("@r1\nACGT\n+\nIIII\n\n@r2\nACGT\n+\nIIII\n", False, id="blank-line"),
    pytest.param("@r1\nACGT\n+\nIII\n", True, id="short-quality"),
    pytest.param("r1\nACGT\n+\nIIII\n", True, id="header-without-at"),
    pytest.param("@r1\nACGT\n+\n\x01\x01\x01\x01\n", True, id="quality-below-range"),
])
def test_the_validator_and_the_trimmer_call_the_same_file_the_same_way(tmp_path, content,
                                                                       refused):
    """One file, two answers, was what the user used to get: the check said "broken"
    and the trimmer trimmed it.  Both now read the same bytes the same way — a file
    the validator clears is one the trimmer processes, and a file it flags is one the
    trimmer refuses by name."""
    path = tmp_path / "reads.fq"
    path.write_text(content, encoding="utf-8")
    errors = InputValidator().validate_fastq(content)

    if refused:
        assert errors, content
        with pytest.raises(ValueError, match=r"^malformed FASTQ in "):
            tr.trim_fastq(str(path), str(tmp_path / "out.fq"), min_length=1)
    else:
        assert errors == [], errors
        report = tr.trim_fastq(str(path), str(tmp_path / "out.fq"), min_length=1)
        assert report.total_reads == content.count('\n') // 4, report.message
