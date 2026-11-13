"""``@validate_range`` — the numeric contract and the bounds configuration.

A range check is only worth what it refuses, and the old version refused too
little: ``float('nan')`` passed a 0-1 bound because every comparison with NaN is
False, a keyword ``True`` was treated as 1 while a positional one was skipped, and
``min_val=5, max_val=1`` was accepted as a configuration that rejects everything.
Which parameter the decorator picks is the subject of
`test_validators_argument_targeting.py`.
"""
import numpy as np
import pytest

from biosuite.core import validators as v


@pytest.fixture
def zero_to_one():
    """``@validate_range(min_val=0.0, max_val=1.0)`` over an identity function."""
    @v.validate_range(min_val=0.0, max_val=1.0)
    def set_threshold(value):
        return value

    return set_threshold


def test_the_bounds_are_inclusive(zero_to_one):
    assert zero_to_one(0.0) == 0.0
    assert zero_to_one(1.0) == 1.0
    assert zero_to_one(0.25) == 0.25


def test_a_value_below_the_minimum_is_reported_with_both_numbers(zero_to_one):
    with pytest.raises(ValueError) as excinfo:
        zero_to_one(-0.1)
    assert str(excinfo.value) == (
        "Value in argument 'value' of set_threshold() (-0.1) is below minimum (0.0)"
    )


def test_a_value_above_the_maximum_is_reported_with_both_numbers(zero_to_one):
    with pytest.raises(ValueError) as excinfo:
        zero_to_one(1.5)
    assert str(excinfo.value) == (
        "Value in argument 'value' of set_threshold() (1.5) is above maximum (1.0)"
    )


def test_an_integer_is_written_without_a_trailing_point():
    @v.validate_range(min_val=1)
    def count(value):
        return value

    with pytest.raises(ValueError) as excinfo:
        count(0)
    assert str(excinfo.value) == (
        "Value in argument 'value' of count() (0) is below minimum (1)"
    )


@pytest.mark.parametrize("value, message", [
    pytest.param(float('nan'), "argument 'value' of set_threshold() must be a finite "
                               "number, got nan.", id="nan"),
    pytest.param(float('inf'), "argument 'value' of set_threshold() must be a finite "
                               "number, got inf.", id="infinity"),
    pytest.param(float('-inf'), "argument 'value' of set_threshold() must be a finite "
                                "number, got -inf.", id="negative-infinity"),
    pytest.param(True, "argument 'value' of set_threshold() must be a number, got bool.",
                 id="bool"),
    pytest.param(False, "argument 'value' of set_threshold() must be a number, got bool.",
                 id="false"),
    pytest.param('0.5', "argument 'value' of set_threshold() must be a number, got str.",
                 id="string"),
    pytest.param(1j, "argument 'value' of set_threshold() must be a number, got complex.",
                 id="complex"),
    pytest.param([0.5], "argument 'value' of set_threshold() must be a number, got list.",
                 id="list"),
])
def test_a_number_that_is_not_a_finite_number_is_refused(zero_to_one, value, message):
    """Measured on the old code: ``set_threshold(float('nan'))`` returned NaN for a
    0-1 bound, because NaN compares False against both of them, and the caller got
    a threshold that made every downstream comparison False as well."""
    with pytest.raises(ValueError) as excinfo:
        zero_to_one(value)
    assert str(excinfo.value) == message


def test_numpy_scalars_are_numbers_and_their_type_is_not_claimed_in_the_message():
    @v.validate_range(min_val=0, max_val=1)
    def ratio(value):
        return value

    assert ratio(np.float32(0.5)) == 0.5
    assert ratio(np.int64(1)) == 1
    with pytest.raises(ValueError) as excinfo:
        ratio(np.float32(1.5))
    assert str(excinfo.value) == (
        "Value in argument 'value' of ratio() (1.5) is above maximum (1)"
    )


def test_one_sided_and_unbounded_ranges_check_only_what_they_declare():
    @v.validate_range(min_val=0)
    def positive(value):
        return value

    @v.validate_range()
    def anything_finite(value):
        return value

    assert positive(10 ** 9) == 10 ** 9
    with pytest.raises(ValueError) as excinfo:
        positive(-1)
    assert str(excinfo.value) == (
        "Value in argument 'value' of positive() (-1) is below minimum (0)"
    )
    assert anything_finite(-1e9) == -1e9
    with pytest.raises(ValueError) as excinfo:
        anything_finite(float('nan'))
    assert "must be a finite number, got nan." in str(excinfo.value)


@pytest.mark.parametrize("bounds, message", [
    pytest.param(dict(min_val=5.0, max_val=1.0),
                 "validate_range needs min_val <= max_val, got min_val=5.0 and "
                 "max_val=1.0; no number could satisfy both.", id="inverted"),
    pytest.param(dict(min_val='0'),
                 "validate_range min_val must be a finite number or None, got '0'.",
                 id="string-bound"),
    pytest.param(dict(max_val=float('nan')),
                 "validate_range max_val must be a finite number or None, got nan.",
                 id="nan-bound"),
    pytest.param(dict(min_val=True),
                 "validate_range min_val must be a finite number or None, got True.",
                 id="bool-bound"),
])
def test_an_impossible_or_misshapen_interval_is_refused_when_the_decorator_is_built(
        bounds, message):
    """Measured on the old code: ``min_val=5, max_val=1`` decorated happily and
    then rejected every number it was given, so a swapped pair of arguments looked
    like a data problem in whichever function it decorated."""
    with pytest.raises(ValueError) as excinfo:
        v.validate_range(**bounds)
    assert str(excinfo.value) == message
