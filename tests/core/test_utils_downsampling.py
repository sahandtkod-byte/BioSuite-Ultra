"""`maybe_downsample` — how a plot decides which points to draw.

The point of this helper is that a picture of 5,000 points and a picture of 50,000
show the same curve, so the two ways it used to fail matter: a `max_points` of `2.9`
was quietly truncated to 2, and a `downsample_threshold` of `null` in the config file
raised `TypeError: int() argument must be ... not 'NoneType'` from inside every plot.
"""
import logging

import numpy as np
import pytest

from biosuite.core import utils


def test_a_series_below_the_limit_is_handed_back_as_it_came():
    x, y = utils.maybe_downsample([1, 2, 3], [4, 5, 6], max_points=5)
    assert (x, y) == ([1, 2, 3], [4, 5, 6])

    array = np.arange(3)
    x, y = utils.maybe_downsample(array, array, max_points=5)
    assert x is array and y is array, "nothing to drop means nothing to copy"


def test_both_endpoints_survive_and_the_stride_is_even():
    x, y = utils.maybe_downsample(list(range(100)), list(range(100)), max_points=9)
    assert x[0] == 0 and x[-1] == 99
    assert len(x) == 9
    assert list(x) == sorted(set(x)), "indices are unique and in order"


def test_the_input_type_is_the_type_that_comes_back():
    x, y = utils.maybe_downsample(tuple(range(50)), list(range(50)), max_points=5)
    assert isinstance(x, tuple) and isinstance(y, list)
    assert len(x) == len(y)

    series = __import__('pandas').Series(np.arange(50.0))
    x, y = utils.maybe_downsample(series, series, max_points=5)
    assert isinstance(x, __import__('pandas').Series), \
        "a Series is sliced by position and stays a Series"
    assert len(x) == 5 and x.iloc[0] == 0.0

    array = np.arange(50.0)
    x, _ = utils.maybe_downsample(array, array, max_points=5)
    assert isinstance(x, np.ndarray) and x is not array


def test_a_sequence_that_is_none_of_those_kinds_is_still_drawable():
    """`range`, `array('d', …)` and friends have no positional indexer of their own;
    they come back as an array of the chosen points."""
    x, y = utils.maybe_downsample(range(50), range(50), max_points=4)
    assert isinstance(x, np.ndarray) and list(x) == [0, 16, 33, 49]
    assert list(y) == [0, 16, 33, 49]


@pytest.mark.parametrize("bad, message", [
    pytest.param(2.9, "max_points must be an integer of at least 2, got 2.9 (float); "
                      "a fractional value would be truncated, not rounded.",
                 id="fraction"),
    pytest.param('50', "max_points must be an integer of at least 2, got '50' (str); "
                       "a fractional value would be truncated, not rounded.",
                 id="string"),
    pytest.param(np.float64(1.5), "max_points must be an integer of at least 2, got "
                                  "np.float64(1.5) (float64); a fractional value would "
                                  "be truncated, not rounded.", id="numpy-float"),
])
def test_a_point_budget_that_is_not_an_integer_is_refused(bad, message):
    """`None` is the one exception: it means "use the configured threshold", which
    the config tests below cover."""
    with pytest.raises(ValueError) as excinfo:
        utils.maybe_downsample(list(range(10)), list(range(10)), max_points=bad)
    assert str(excinfo.value) == message


def test_a_boolean_is_not_a_point_budget():
    """`int(True)` is 1, which used to reach the "at least 2" check as a confusing
    message about the number 1."""
    with pytest.raises(ValueError) as excinfo:
        utils.maybe_downsample(list(range(10)), list(range(10)), max_points=True)
    assert str(excinfo.value) == (
        "max_points must be an integer of at least 2, got True (bool); a fractional "
        "value would be truncated, not rounded.")


@pytest.mark.parametrize("budget", [0, 1, -3])
def test_a_budget_below_two_points_cannot_draw_a_curve(budget):
    with pytest.raises(ValueError) as excinfo:
        utils.maybe_downsample(list(range(10)), list(range(10)), max_points=budget)
    assert str(excinfo.value) == f"max_points must be at least 2, got {budget}"


def test_ragged_series_are_reported_with_both_lengths():
    with pytest.raises(ValueError) as excinfo:
        utils.maybe_downsample([1, 2, 3], [1, 2], max_points=2)
    assert str(excinfo.value) == "x and y must be the same length, got 3 and 2"


@pytest.mark.parametrize("stored, kept", [
    pytest.param(None, 5000, id="null-in-config"),
    pytest.param('5000', 5000, id="string-in-config"),
    pytest.param(2.5, 5000, id="fraction-in-config"),
    pytest.param(True, 5000, id="bool-in-config"),
    pytest.param(10, 10, id="a_usable_number"),
])
def test_a_bad_configured_threshold_is_reported_and_the_default_used(
        monkeypatch, caplog, stored, kept):
    """The threshold comes from `biosuite_config.json`, which a user can hand-edit;
    a `null` there used to raise `TypeError` inside every plot."""
    monkeypatch.setitem(utils.config, 'downsample_threshold', stored)
    big = list(range(30))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        x, y = utils.maybe_downsample(big, big)
    assert len(x) == min(kept, 30) == len(y)
    notes = [r.getMessage() for r in caplog.records if r.name == 'biosuite.core.utils']
    if stored == 10:
        assert notes == ["Downsampled from 30 to 10 points."]
    else:
        assert notes[0] == (f"Config 'downsample_threshold' is {stored!r}, not an "
                            f"integer; using 5000 points.")
