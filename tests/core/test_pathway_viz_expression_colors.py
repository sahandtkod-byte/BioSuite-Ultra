"""The expression colour scale on a pathway node.

Colour is how a reader tells an up-regulated node from a down-regulated one at a
glance, so the mapping has to be continuous, ordered, and refuse values it cannot
place rather than painting them arbitrarily.
"""
import re

import pytest

from biosuite.core.pathway_viz import EXPRESSION_ANCHORS, PathwayMap, PathwayNode


def _colour(value, vmin=-2, vmax=2):
    node = PathwayNode("x", "x")
    node.set_expression(value, vmin=vmin, vmax=vmax)
    return node.color


def test_every_colour_the_scale_produces_is_a_six_digit_hex():
    for value in (-10, -2, -1, 0, 1, 2, 10):
        assert re.fullmatch(r"#[0-9a-f]{6}", _colour(value)), value


def test_the_two_ends_of_the_scale_are_the_anchor_colours():
    assert _colour(-2) == "#1e50dc"
    assert _colour(2) == "#e61e14"
    assert EXPRESSION_ANCHORS[0] == (30, 80, 220)
    assert EXPRESSION_ANCHORS[-1] == (230, 30, 20)


def test_no_change_in_expression_is_neutral_rather_than_a_hue():
    """The midpoint used to be a saturated pale yellow (#e6e6aa).

    A colour the reader cannot place is read as a small effect, so the node that
    did not change looks like it did.
    """
    assert _colour(0) == "#ebebeb"


def test_the_scale_is_continuous_across_the_midpoint():
    """Measured before the fix: -0.01 was #e5b3aa and 0.0 was #e6e6aa.

    Two neighbouring values differed by 77 units of green — a visible band edge
    drawn across the middle of every map at exactly the point that means "nothing
    happened".
    """
    below, at, above = _colour(-0.01), _colour(0.0), _colour(0.01)
    channels = [tuple(int(pair, 16) for pair in (c[1:3], c[3:5], c[5:7]))
                for c in (below, at, above)]
    steps = [max(abs(a[k] - b[k]) for k in range(3))
             for a, b in zip(channels, channels[1:])]
    assert max(steps) <= 2, (below, at, above)


def test_the_scale_gets_warmer_monotonically_as_expression_rises():
    """blue minus red falls strictly across the range, and no two values collide."""
    colours = [_colour(v / 10) for v in range(-20, 21)]
    assert len(set(colours)) == len(colours), "the ramp is flat somewhere"
    dominance = [int(c[5:7], 16) - int(c[1:3], 16) for c in colours]
    assert all(high > low for high, low in zip(dominance, dominance[1:])), dominance


def test_values_outside_the_range_clamp_to_the_end_they_belong_to():
    assert _colour(-10) == _colour(-2)
    assert _colour(10) == _colour(2)


def test_the_raw_value_is_stored_even_when_the_colour_clamps():
    node = PathwayNode("x", "x")
    node.set_expression(3.0, vmin=-1, vmax=1)
    assert node.expression == 3.0
    assert node.color == "#e61e14"


def test_a_custom_range_is_honoured():
    assert _colour(5.0, vmin=0, vmax=10) == "#ebebeb"
    assert _colour(0.0, vmin=0, vmax=10) == "#1e50dc"


def test_a_node_untouched_by_the_scale_holds_no_value_and_no_colour():
    node = PathwayNode("x", "x")
    assert (node.expression, node.color) == (None, None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_expression_is_refused_instead_of_crashing_mid_paint(value):
    """Measured before the fix: NaN raised `ValueError: cannot convert float NaN to
    integer` from inside the colour arithmetic, and inf painted the node without
    recording anything about it."""
    with pytest.raises(ValueError) as exc:
        _colour(value)
    assert f"must be a finite number, got {value!r}" in str(exc.value)


@pytest.mark.parametrize("value", [None, "1.5", [1.5], True])
def test_an_expression_that_is_not_a_number_is_a_named_type_error(value):
    with pytest.raises(TypeError) as exc:
        _colour(value)
    assert "expression value must be a number" in str(exc.value)


@pytest.mark.parametrize("bounds", [(0.0, 0.0), (2.0, -2.0), (5, 5)])
def test_a_scale_without_a_positive_span_is_refused(bounds):
    """Equal bounds divided by zero; reversed ones painted the opposite hue.

    A caller that hands over `vmin=2, vmax=-2` would otherwise get a map whose
    colours mean the reverse of what the numbers say.
    """
    with pytest.raises(ValueError) as exc:
        _colour(1.0, vmin=bounds[0], vmax=bounds[1])
    assert "the colour scale needs vmax above vmin" in str(exc.value)


def test_setting_expression_through_the_map_reaches_only_real_nodes():
    pm = PathwayMap("t")
    pm.add_node("a", "A")
    pm.set_expression({"a": 1.0, "ghost": -1.0})
    assert pm.nodes["a"].expression == 1.0
    assert "ghost" not in pm.nodes


def test_the_map_forwards_the_bounds_it_was_given():
    pm = PathwayMap("t")
    pm.add_node("a", "A")
    pm.set_expression({"a": 10.0}, vmin=0, vmax=10)
    assert pm.nodes["a"].color == "#e61e14"
    with pytest.raises(ValueError):
        pm.set_expression({"a": 1.0}, vmin=1, vmax=1)
