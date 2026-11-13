"""An UpSet plot refuses a table or a limit it cannot draw.

An empty mapping has no set names to print, a limit below one keeps nothing, and
a value that is not a set cannot be intersected.  A supplied axes cannot hold
both UpSet panels, so asking for one with data is refused instead of silently
drawing the diagram in a second figure the caller never sees.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.upset_plots import (
    compute_set_statistics,
    compute_upset_matrix,
    plot_set_sizes,
    plot_upset,
    upset_from_sets,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_an_empty_mapping_has_no_upset_to_draw():
    with pytest.raises(ValueError, match="at least one set"):
        plot_upset({})
    assert plt.get_fignums() == []


def test_an_empty_mapping_has_no_sizes_to_draw():
    with pytest.raises(ValueError, match="at least one set"):
        plot_set_sizes({})


@pytest.mark.parametrize("limit", [0, -2])
def test_a_set_limit_below_one_keeps_nothing(limit):
    with pytest.raises(ValueError, match="max_sets must be at least 1"):
        plot_upset({"A": {1}}, max_sets=limit)


def test_a_set_limit_that_is_not_a_whole_number_is_named():
    with pytest.raises(ValueError, match="max_sets must be a whole number, got '5'"):
        plot_upset({"A": {1}}, max_sets="5")


def test_a_single_axes_is_refused_when_there_is_a_diagram_to_draw():
    outer, ax = plt.subplots()

    with pytest.raises(ValueError, match="cannot draw both"):
        plot_upset({"A": {1, 2}, "B": {2, 3}}, ax=ax)

    assert len(outer.axes) == 1, "the caller's figure gained no panel"
    assert len(ax.patches) == 0 and len(ax.texts) == 0, \
        "nothing was drawn on the supplied axes"
    assert len(plt.get_fignums()) == 1, "no second figure was created"


def test_a_value_that_is_not_a_set_is_named():
    with pytest.raises(TypeError, match="set 'A' must be a set of elements, got list"):
        compute_upset_matrix({"A": [1, 2]})


def test_the_statistics_refuse_a_value_that_is_not_a_set():
    with pytest.raises(TypeError, match="set 'A' must be a set of elements, got list"):
        compute_set_statistics({"A": [1, 2]})


def test_a_mapping_that_is_not_a_mapping_is_named():
    with pytest.raises(ValueError, match="must be a mapping of set names to sets"):
        compute_upset_matrix([{"A"}])


def test_a_table_past_the_set_budget_is_refused():
    sets = {f"S{i}": {i} for i in range(17)}

    with pytest.raises(ValueError, match="131,071") as excinfo:
        compute_upset_matrix(sets)

    assert "at most 16" in str(excinfo.value)


def test_the_plot_refuses_a_table_past_the_set_budget():
    sets = {f"S{i}": {i} for i in range(17)}

    with pytest.raises(ValueError, match="at most 16"):
        plot_upset(sets, max_sets=17)


def test_the_full_budget_of_sixteen_sets_is_still_accepted():
    labels, matrix, counts = compute_upset_matrix({f"S{i}": {i} for i in range(16)})

    assert len(labels) == 16
    assert len(matrix) == len(counts) == 16


def test_an_empty_table_is_empty_rather_than_an_error():
    assert compute_upset_matrix({}) == ([], [], [])


def test_statistics_of_no_sets_are_zero():
    assert compute_set_statistics({}) == {
        'sizes': {},
        'total_union': 0,
        'total_intersection': 0,
        'pairwise_jaccard': {},
        'unique_per_set': {},
    }


def test_two_empty_sets_have_no_jaccard():
    stats = compute_set_statistics({"A": set(), "B": set()})

    assert stats['pairwise_jaccard'][('A', 'B')] == 0.0


def test_a_convenience_constructor_needs_a_set():
    with pytest.raises(ValueError, match="at least one set"):
        upset_from_sets()


def test_a_short_name_list_is_refused():
    with pytest.raises(ValueError, match="one name per set"):
        upset_from_sets({1}, {2}, names=["only"])


def test_a_long_name_list_is_refused():
    with pytest.raises(ValueError, match="one name per set"):
        upset_from_sets({1}, names=["one", "two"])


def test_a_repeated_set_name_is_refused():
    with pytest.raises(ValueError, match="set names repeat"):
        upset_from_sets({1}, {2}, names=["S", "S"])


def test_a_name_that_is_not_a_string_is_reported_by_position():
    with pytest.raises(ValueError, match="name 1 is not a non-empty string: None"):
        upset_from_sets({1}, names=[None])
