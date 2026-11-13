"""Sankey links the drawer can actually place.

``draw_sankey`` is public, and three shapes used to reach it with no answer:
no links at all died in ``max(values)`` with numpy's "max() iterable argument is
empty"; a link index outside the label list died in ``labels[i]`` with a bare
``IndexError``; and a table of all-zero flows divided by itself, drawing every
link at ``linewidth = nan``.  ``sources``/``targets``/``values`` of different
lengths zipped quietly down to the shortest, so half the flows vanished from a
diagram whose title promised all of them.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import biological_plots as bp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _link_widths(ax):
    from matplotlib.patches import ConnectionPatch

    return [round(patch.get_linewidth(), 3) for patch in ax.patches
            if isinstance(patch, ConnectionPatch)]


def test_a_normal_flow_set_draws_one_link_scaled_to_the_biggest():
    ax = bp.draw_sankey(np.array(['gut', 'blood']), np.array([0]), np.array([1]),
                        np.array([5]))

    assert _link_widths(ax) == [10.0], "2 + 8 * 5/5"
    assert len(ax.patches) == 3, "one rectangle per node plus the link"
    assert ax.get_title() == 'Sankey Diagram (internal)'


def test_two_flows_between_the_same_pair_both_get_drawn():
    ax = bp.draw_sankey(np.array(['a', 'b']), np.array([0, 0]), np.array([1, 1]),
                        np.array([2, 8]))

    assert _link_widths(ax) == [4.0, 10.0]


def test_no_links_is_drawn_as_an_empty_diagram_instead_of_crashing():
    """The old code reached ``max()`` on an empty sequence before it could find out
    that there was nothing to lay out."""
    ax = bp.draw_sankey(np.array(['a', 'b']), np.array([], dtype=int),
                        np.array([], dtype=int), np.array([]))

    assert ax.get_title() == 'Sankey Diagram (no flows to draw)', ax.get_title()
    assert len(ax.patches) == 0, "nothing to place, so nothing is placed"


def test_flows_that_are_all_zero_draw_hairlines_and_say_so(capsys):
    ax = bp.draw_sankey(np.array(['a', 'b']), np.array([0]), np.array([1]),
                        np.array([0]))

    assert _link_widths(ax) == [2.0], "the minimum width, not nan"
    assert "every flow is zero" in capsys.readouterr().out


@pytest.mark.parametrize("sources, targets, values, counts", [
    ([0, 1], [0], [1.0], '2 sources, 1 targets and 1 values'),
    ([0], [0, 1], [1.0], '1 sources, 2 targets and 1 values'),
    ([0], [0], [1.0, 2.0], '1 sources, 1 targets and 2 values'),
])
def test_arrays_of_different_lengths_are_named_with_their_lengths(
        sources, targets, values, counts):
    with pytest.raises(ValueError, match='one source, target and value per link'):
        bp.draw_sankey(np.array(['a', 'b']), np.array(sources), np.array(targets),
                       np.array(values))

    try:
        bp.draw_sankey(np.array(['a', 'b']), np.array(sources), np.array(targets),
                       np.array(values))
    except ValueError as exc:
        assert counts in str(exc), str(exc)
    else:
        pytest.fail("the mismatch was accepted")


def test_an_empty_label_list_is_refused_before_any_index_is_read():
    with pytest.raises(ValueError, match='no node labels to name a link by'):
        bp.draw_sankey(np.array([]), np.array([0]), np.array([0]), np.array([1.0]))


def test_a_link_endpoint_outside_the_labels_names_the_index_and_the_range():
    with pytest.raises(ValueError, match='5 labels') as info:
        bp.draw_sankey(np.arange(5).astype(str), np.array([0, 9]), np.array([1, 1]),
                       np.array([1.0, 2.0]))

    text = str(info.value)
    assert 'indices 0..4' in text, text
    assert 'e.g. index 9' in text, text


def test_a_negative_endpoint_is_outside_the_labels_too():
    with pytest.raises(ValueError, match='link end\\(s\\) outside'):
        bp.draw_sankey(np.array(['a', 'b']), np.array([-1]), np.array([1]),
                       np.array([3.0]))
