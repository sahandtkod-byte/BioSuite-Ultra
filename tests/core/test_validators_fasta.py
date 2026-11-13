"""`InputValidator.validate_fasta` — record structure, alphabets and line numbers.

Two properties carry this file.  A FASTA report has to point at the line the user
sees in the editor, so leading blank lines must not shift the numbers — the old
implementation stripped the content before counting and reported a problem on real
line 3 as "Line 1".  And the sequence check needs to know what kind of file it is
reading: with no `seq_type`, every protein FASTA in the package was a list of
"invalid characters", although `InputValidator` documented amino-acid validation.
"""
import pytest

from biosuite.core.validators import InputValidator


@pytest.fixture
def validator():
    return InputValidator()


LONG_HEADER = '>' + 'h' * 60


# ── the shape of a FASTA file ───────────────────────────────────────────────

def test_a_clean_multi_record_file_needs_no_complaints(validator):
    assert validator.validate_fasta(">s1\nACGT\n>s2\nacgtnn\n") == []
    assert validator.validate_fasta(">s1 description line\nACGT\n") == []


def test_a_sequence_split_over_several_lines_is_one_sequence(validator):
    assert validator.validate_fasta(">s1\nACG\nTTT\n\nGGG\n") == []


@pytest.mark.parametrize("content", ["", "   ", "\n\n", None])
def test_content_that_holds_nothing_is_not_a_file(validator, content):
    assert validator.validate_fasta(content) == ["FASTA content is empty"]


def test_a_file_without_headers_is_reported_once_and_stops_there(validator):
    assert validator.validate_fasta("ACGT\nTTTT\n") == [
        "No FASTA headers found (lines starting with '>')"
    ]


def test_the_first_content_line_has_to_be_a_header(validator):
    assert validator.validate_fasta("ACGT\n>s1\nACGT\n") == [
        "Line 1: First non-empty line must be a header (starting with '>')"
    ]


def test_a_header_line_with_no_name_is_an_error(validator):
    assert validator.validate_fasta(">\nACGT\n") == ["Line 1: Empty header"]
    assert validator.validate_fasta(">   \nACGT\n") == ["Line 1: Empty header"]


def test_a_header_without_sequence_data_is_named_by_its_own_text(validator):
    assert validator.validate_fasta(">s1\n\n>s2\nACGT\n") == [
        "Line 1: No sequence data after header '>s1'"
    ]
    assert validator.validate_fasta(">s1\nACGT\n>s2\n") == [
        "Line 3: No sequence data after header '>s2'"
    ]


# ── line numbers are the file's, not the stripped string's ──────────────────

def test_leading_blank_lines_do_not_move_the_reported_line_numbers(validator):
    """Measured on the old code: the content was `strip()`ed before it was split,
    so a header on real line 3 was reported as "Line 1" and the user edited the
    wrong line."""
    assert validator.validate_fasta("\n\n>\nACGT\n") == ["Line 3: Empty header"]
    assert validator.validate_fasta("\nACGT\n>s1\nACGT\n") == [
        "Line 2: First non-empty line must be a header (starting with '>')"
    ]
    assert validator.validate_fasta("\n\n\n>s1\n\n") == [
        "Line 4: No sequence data after header '>s1'"
    ]


# ── the alphabets ───────────────────────────────────────────────────────────

def test_dna_is_the_default_and_the_check_names_the_alphabet(validator):
    assert validator.validate_fasta(">s1\nACGU\n") == [
        "Header '>s1': invalid DNA characters ['U']"
    ]
    assert validator.validate_fasta(">s1\nACGTQ\n") == [
        "Header '>s1': invalid DNA characters ['Q']"
    ]


def test_ambiguity_codes_and_gaps_pass_a_nucleotide_file(validator):
    assert validator.validate_fasta(">s1\nACGTRYSWKMBDHVN\n") == []
    assert validator.validate_fasta(">s1\nACG-TA\n") == []
    assert validator.validate_fasta(">s1\nACG..TA\n") == []
    assert validator.validate_fasta(">s1\nACGU\n", seq_type='rna') == []


def test_a_protein_file_needs_to_say_so_and_then_passes(validator):
    """Measured on the old code: the DNA-only check answered a UniProt record with
    "invalid characters ['K', 'L', 'M', 'S', 'V']" — the protein sequence was read
    as a broken nucleotide sequence, and there was no way to ask for the other
    alphabet although the class advertised amino-acid validation."""
    content = ">sp|P12345|BRCA2_HUMAN\nMKVKGTLS*\n"
    assert validator.validate_fasta(content) == [
        "Header '>sp|P12345|BRCA2_HUMAN': invalid DNA characters ['L']"
    ]
    assert validator.validate_fasta(content, seq_type='amino_acid') == []


def test_the_interior_of_a_sequence_line_is_not_whitespace_tolerant(validator):
    """`validate_sequence` pastes cleanly; a file keeps its spaces because a space
    inside a FASTA sequence line is not something a reader should guess about."""
    assert validator.validate_fasta(">s1\nACG T\n") == [
        "Header '>s1': invalid DNA characters [' ']"
    ]


def test_every_record_is_checked_and_the_header_is_quoted_in_full_up_to_40(validator):
    assert validator.validate_fasta(">a\nACGTQ\n>b\nACGTX\n") == [
        "Header '>a': invalid DNA characters ['Q']",
        "Header '>b': invalid DNA characters ['X']",
    ]
    errors = validator.validate_fasta(f"{LONG_HEADER}\nACGTZQ\n")
    assert errors == [
        f"Header '{LONG_HEADER[:40]}': invalid DNA characters ['Q', 'Z']"
    ]
    assert len(errors[0].split("': invalid")[0]) == len("Header '") + 40


def test_a_sequence_type_the_caller_guessed_wrong_is_reported_not_guessed(validator):
    assert validator.validate_fasta(">s1\nACGT\n", seq_type='protein') == [
        "Unknown sequence type: 'protein'; use 'dna', 'rna' or 'amino_acid'."
    ]
    assert validator.validate_fasta(">s1\nACGT\n", seq_type=None) == [
        "Unknown sequence type: None; use 'dna', 'rna' or 'amino_acid'."
    ]
