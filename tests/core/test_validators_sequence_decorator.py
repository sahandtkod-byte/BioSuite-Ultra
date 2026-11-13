"""``@validate_sequence`` — length, alphabet and configuration refusals.

The decorator is the module's own contract for "this argument is a nucleotide
sequence", so both halves are pinned here: what a bad sequence is reported as, and
what happens when the *decorator itself* is misconfigured.  Which parameter the
decorator picks is the subject of `test_validators_argument_targeting.py`.
"""
import pytest

from biosuite.core import validators as v


@pytest.fixture
def at_least_two():
    """``@validate_sequence(min_length=2)`` over an identity function."""
    @v.validate_sequence(min_length=2)
    def identity(seq):
        return seq

    return identity


def test_a_valid_sequence_reaches_the_function_unchanged(at_least_two):
    assert at_least_two("ACGT") == "ACGT"
    assert at_least_two("acgt") == "acgt"


def test_a_short_sequence_names_its_length_and_the_minimum(at_least_two):
    with pytest.raises(ValueError) as excinfo:
        at_least_two("A")
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of identity() is too short: 1 < 2 (min_length=2)"
    )


def test_the_empty_string_is_reported_by_its_own_length(at_least_two):
    with pytest.raises(ValueError) as excinfo:
        at_least_two("")
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of identity() is too short: 0 < 2 (min_length=2)"
    )


def test_the_offending_characters_and_the_alphabet_are_listed_in_order(at_least_two):
    with pytest.raises(ValueError) as excinfo:
        at_least_two("ACGXQ")
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of identity() contains invalid characters: ['Q', 'X']. "
        "Allowed characters: ['A', 'C', 'G', 'N', 'T']"
    )


def test_the_character_match_is_case_insensitive():
    @v.validate_sequence(allowed_chars='atc')
    def lower(seq):
        return seq

    assert lower('AcT') == 'AcT'
    with pytest.raises(ValueError) as excinfo:
        lower('ATG')
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of lower() contains invalid characters: ['G']. "
        "Allowed characters: ['A', 'C', 'T']"
    )


def test_the_default_alphabet_is_the_five_DNA_codes_and_the_default_minimum_is_one():
    @v.validate_sequence()
    def defaults(seq):
        return seq

    assert defaults('ATCGN') == 'ATCGN'
    with pytest.raises(ValueError) as excinfo:
        defaults('')
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of defaults() is too short: 0 < 1 (min_length=1)"
    )
    with pytest.raises(ValueError) as excinfo:
        defaults('U')
    assert "invalid characters: ['U']" in str(excinfo.value)


def test_an_empty_alphabet_forbids_every_character_not_every_check():
    @v.validate_sequence(min_length=0, allowed_chars='')
    def nothing_allowed(seq):
        return seq

    assert nothing_allowed("") == ""
    with pytest.raises(ValueError) as excinfo:
        nothing_allowed("X")
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of nothing_allowed() contains invalid characters: "
        "['X']. Allowed characters: []"
    )


@pytest.mark.parametrize("min_length, error, message", [
    pytest.param('5', ValueError,
                 "validate_sequence min_length must be a non-negative integer, got '5'.",
                 id="string"),
    pytest.param(1.5, ValueError,
                 "validate_sequence min_length must be a non-negative integer, got 1.5.",
                 id="float"),
    pytest.param(-1, ValueError,
                 "validate_sequence min_length must be a non-negative integer, got -1.",
                 id="negative"),
    pytest.param(True, ValueError,
                 "validate_sequence min_length must be a non-negative integer, got True.",
                 id="bool"),
    pytest.param(None, ValueError,
                 "validate_sequence min_length must be a non-negative integer, got None.",
                 id="none"),
])
def test_a_minimum_that_is_not_a_length_is_refused_when_the_decorator_is_built(
        min_length, error, message):
    """Before this check, ``min_length='5'`` reached ``len(seq) < min_length`` and
    raised ``TypeError: '<' not supported between instances of 'int' and 'str'``
    from inside the validator, far from the misconfigured decorator."""
    with pytest.raises(error) as excinfo:
        v.validate_sequence(min_length=min_length)
    assert str(excinfo.value) == message


@pytest.mark.parametrize("allowed_chars, message", [
    pytest.param(None, "validate_sequence allowed_chars must be a string of characters, "
                       "got NoneType.", id="none"),
    pytest.param(['A', 'T'], "validate_sequence allowed_chars must be a string of "
                             "characters, got list.", id="list"),
    pytest.param(1234, "validate_sequence allowed_chars must be a string of characters, "
                       "got int.", id="number"),
])
def test_an_alphabet_that_is_not_a_string_of_characters_is_refused(allowed_chars, message):
    """A list of codes would silently become ``set([...])``, so it has to be said
    in the caller's terms: the alphabet is a string of characters."""
    with pytest.raises(TypeError) as excinfo:
        v.validate_sequence(allowed_chars=allowed_chars)
    assert str(excinfo.value) == message
