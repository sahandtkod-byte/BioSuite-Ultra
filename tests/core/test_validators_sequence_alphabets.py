"""The alphabets `InputValidator` validates a sequence against.

`biosuite/core/sequence.py` defines the package's nucleotide alphabet —
`IUPAC_NUCLEOTIDES`, ambiguity codes and alignment gaps included — and
`InputValidator.VALID_DNA` used to be a five-character set beside it.  Two answers
in one package is how a soft-masked or aligned FASTA gets read as an error by one
module and as data by another, so the two are pinned against each other here.
"""
import pytest

from biosuite.core import sequence as sq
from biosuite.core.validators import InputValidator, default_validator


@pytest.fixture
def validator():
    return InputValidator()


# ── the alphabets ───────────────────────────────────────────────────────────

def test_the_nucleotide_sets_are_the_ones_the_package_already_uses(validator):
    """Not a copy: the same characters `sequence.IUPAC_NUCLEOTIDES` accepts, with
    the DNA set keeping T and the RNA set keeping U."""
    iupac = {c.upper() for c in sq.IUPAC_NUCLEOTIDES}
    assert set(validator.VALID_DNA) == iupac - {'U'}
    assert set(validator.VALID_RNA) == iupac - {'T'}


def test_ambiguity_codes_and_gaps_are_valid_nucleotides(validator):
    assert validator.validate_sequence('ACGTRYSWKMBDHVN') == []
    assert validator.validate_sequence('ACGTR') == []
    assert validator.validate_sequence('ACG-TA') == []
    assert validator.validate_sequence('ACGU', seq_type='rna') == []
    assert validator.validate_sequence('ACG-N', seq_type='rna') == []


def test_a_base_from_the_other_alphabet_is_still_the_wrong_base(validator):
    assert validator.validate_sequence('ACGTU') == ["Invalid DNA characters: ['U']"]
    assert validator.validate_sequence('ACGT', seq_type='rna') == \
        ["Invalid RNA characters: ['T']"]


def test_the_amino_acid_set_is_the_IUPAC_one_including_stop_and_unknown(validator):
    assert validator.validate_sequence('MKVKGTLS', seq_type='amino_acid') == []
    assert validator.validate_sequence('ACDEFGHIKLMNPQRSTVWY*', seq_type='amino_acid') == []
    # X unknown, B Asx, Z Glx, J selenocysteine-pyrrolysine, O pyrrolysine, U Sec
    assert validator.validate_sequence('XBZJOU', seq_type='amino_acid') == []
    assert validator.validate_sequence('MKV-W', seq_type='amino_acid') == []
    assert validator.validate_sequence('MKV1', seq_type='amino_acid') == \
        ["Invalid amino acid characters: ['1']"]


def test_a_character_that_is_not_in_any_alphabet_is_named(validator):
    assert validator.validate_sequence('ACGT1') == ["Invalid DNA characters: ['1']"]
    assert validator.validate_sequence('acgtx') == ["Invalid DNA characters: ['X']"]


# ── the shape of the input ──────────────────────────────────────────────────

@pytest.mark.parametrize("sequence, message", [
    pytest.param('', "Sequence must be a non-empty string", id="empty"),
    pytest.param(None, "Sequence must be a string, got NoneType.", id="none"),
    pytest.param(123, "Sequence must be a string, got int.", id="number"),
    pytest.param(['A', 'C'], "Sequence must be a string, got list.", id="list"),
    pytest.param(b'ACGT', "Sequence must be a string, got bytes.", id="bytes"),
])
def test_something_that_is_not_a_sequence_string_is_reported_as_such(validator, sequence,
                                                                      message):
    assert validator.validate_sequence(sequence) == [message]


def test_whitespace_is_formatting_and_not_sequence_content(validator):
    assert validator.validate_sequence(" A C G \n T ") == []
    assert validator.validate_sequence('AT\tCG') == []
    assert validator.validate_sequence('ACG\r\nT') == []
    assert validator.validate_sequence('   ') == ['Sequence length 0 is below minimum 1']


def test_the_length_check_counts_the_sequence_not_the_paste(validator):
    assert validator.validate_sequence('AC\nGT', min_length=5) == \
        ['Sequence length 4 is below minimum 5']
    assert validator.validate_sequence('ACGT', min_length=4) == []
    assert validator.validate_sequence('AT XG\n', min_length=10) == [
        'Sequence length 4 is below minimum 10',
        "Invalid DNA characters: ['X']",
    ]


def test_an_empty_string_is_refused_before_any_length_is_considered(validator):
    """`min_length=0` cannot make the empty string a sequence."""
    assert validator.validate_sequence('', min_length=0) == \
        ['Sequence must be a non-empty string']


# ── the two knobs of the method itself ──────────────────────────────────────

@pytest.mark.parametrize("seq_type, expected", [
    pytest.param('dna', [], id="dna"),
    pytest.param('DNA', [], id="upper"),
    pytest.param(' Rna ', [], id="padded-rna"),
    pytest.param('AMINO_ACID', [], id="upper-amino-acid"),
])
def test_the_type_spelling_is_matched_case_insensitively(validator, seq_type, expected):
    """Measured on the old code: `seq_type='DNA'` came back
    "Unknown sequence type: DNA" although the method upper-cases the sequence it
    was given — the sequence was read case-insensitively and its type was not."""
    sequence = 'ACGU' if seq_type.strip().lower() == 'rna' else 'ACGT'
    assert validator.validate_sequence(sequence, seq_type=seq_type) == expected


@pytest.mark.parametrize("seq_type, message", [
    pytest.param('protein', "Unknown sequence type: 'protein'; use 'dna', 'rna' or "
                            "'amino_acid'."),
    pytest.param('nucleotide', "Unknown sequence type: 'nucleotide'; use 'dna', 'rna' or "
                               "'amino_acid'."),
    pytest.param(None, "Unknown sequence type: None; use 'dna', 'rna' or 'amino_acid'."),
    pytest.param(1, "Unknown sequence type: 1; use 'dna', 'rna' or 'amino_acid'."),
])
def test_an_unknown_type_is_refused_with_the_list_of_known_ones(validator, seq_type,
                                                                message):
    assert validator.validate_sequence('ACGT', seq_type=seq_type) == [message]


@pytest.mark.parametrize("min_length, message", [
    pytest.param(None, "min_length must be a non-negative integer, got None."),
    pytest.param('2', "min_length must be a non-negative integer, got '2'."),
    pytest.param(-1, "min_length must be a non-negative integer, got -1."),
    pytest.param(1.5, "min_length must be a non-negative integer, got 1.5."),
    pytest.param(True, "min_length must be a non-negative integer, got True."),
])
def test_a_minimum_that_is_not_a_length_comes_back_as_an_error_not_a_crash(
        validator, min_length, message):
    """Measured on the old code: `validate_sequence('ACGT', min_length=None)`
    raised `TypeError: '<' not supported between instances of 'int' and
    'NoneType'` from a method whose entire contract is a list of error strings."""
    assert validator.validate_sequence('ACGT', min_length=min_length) == [message]


def test_the_module_level_instance_is_the_same_kind_of_validator():
    assert isinstance(default_validator, InputValidator)
    assert default_validator.validate_sequence('ACGT') == []
