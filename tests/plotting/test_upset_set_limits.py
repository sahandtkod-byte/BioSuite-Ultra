"""UpSet edges: too many sets, nothing to intersect, and the helper constructors.

``plot_upset`` keeps the largest sets when the table has more than ``max_sets`` of
them and says so by drawing only what it kept; when a row of the intersection
matrix would be empty for every combination it returns without a figure.  The two
helpers below it default their own names and axes so a caller can hand it plain
sets, and ``plot_set_sizes`` answers with the axes it drew on, not a figure.
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


def test_more_sets_than_the_limit_keeps_the_biggest_ones():
    sets = {f"S{i}": {i, 100 + i} for i in range(12)}
    sets["S0"] = {0, 1, 2, 3, 4, 5}
    sets["S1"] = {1, 2, 3, 4}
    sets["S2"] = {2, 3}

    fig = plot_upset(sets, max_sets=5)

    labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert len(labels) == 5, f"the matrix kept {len(labels)} of 12 sets"
    kept = sorted(sets, key=lambda k: len(sets[k]), reverse=True)[:5]
    assert sorted(labels) == sorted(kept), "the five largest sets are the ones kept"
    bars = fig.axes[0].patches
    assert len(bars) == len(fig.axes[0].texts), "every bar is annotated with its own count"


def test_identical_sets_leave_one_row_for_the_shared_elements():
    """Nothing is exclusive to either set, so the only row is the one marking both."""
    overlapping = {"A": {1, 2}, "B": {1, 2}}

    labels, matrix, counts = compute_upset_matrix(overlapping)
    assert (labels, matrix, counts) == (["A", "B"], [[1, 1]], [2])

    fig = plot_upset(overlapping)
    assert [p.get_height() for p in fig.axes[0].patches] == [2]
    assert len(fig.axes[1].patches) == 2, "one dot per set in the row"


def test_sets_holding_nothing_draw_no_figure():
    assert plt.get_fignums() == []

    assert plot_upset({"A": set(), "B": set()}) is None
    assert plt.get_fignums() == [], "an empty table draws nothing at all"


def test_an_empty_intersection_reports_on_the_axes_it_was_given():
    outer, ax = plt.subplots()

    returned = plot_upset({"A": set(), "B": set()}, ax=ax)

    assert returned is None
    assert [t.get_text() for t in ax.texts] == ["No intersections"]


def test_upset_from_sets_names_the_sets_it_was_not_given():
    fig = upset_from_sets({1, 2, 3}, {3, 4})

    assert fig is not None
    labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert labels == ["Set 1", "Set 2"], labels


def test_upset_from_sets_keeps_the_names_it_was_handed():
    fig = upset_from_sets({1, 2}, {2, 3}, names=["Kidney", "Liver"])

    assert fig is not None
    labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert sorted(labels) == ["Kidney", "Liver"], labels


def test_set_sizes_answers_with_the_axes_it_drew_on():
    sizes = {"A": {1, 2, 3}, "B": {1}, "C": {1, 2}}

    ax = plot_set_sizes(sizes)

    assert isinstance(ax, plt.Axes), "the caller gets an axes, not a figure"
    assert [p.get_height() for p in ax.patches] == [3, 1, 2], "one bar per set"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"], "the bars follow sorted order"
    assert [t.get_text() for t in ax.texts] == ["3", "1", "2"], "each bar says how big it is"


def test_set_sizes_adds_to_the_axes_it_was_handed():
    outer, ax = plt.subplots()

    returned = plot_set_sizes({"A": {1}, "B": {2, 3}}, ax=ax)

    assert returned is ax
    assert len(outer.axes) == 1, "no second axes appears behind the caller's back"


def test_set_statistics_count_the_union_and_the_shared_part():
    stats = compute_set_statistics({"A": {1, 2, 3}, "B": {3, 4}})

    assert stats['sizes'] == {'A': 3, 'B': 2}
    assert stats['total_union'] == 4
    assert stats['total_intersection'] == 1
    assert stats['unique_per_set'] == {'A': 2, 'B': 1}
    assert stats['pairwise_jaccard'] == {('A', 'B'): pytest.approx(1 / 4)}
