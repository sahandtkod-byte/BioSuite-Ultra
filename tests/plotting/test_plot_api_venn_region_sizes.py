"""Venn region sizes are counts, and the sunburst has to add up.

``venn()`` takes the *region* sizes — ``[onlyA, onlyB, both]`` or
``[A, B, C, AB, AC, BC, ABC]`` — and the interactive branch fed them straight
into a ``go.Sunburst`` with ``branchvalues='total'``.  That option means a
parent's own value *is* its total, so the wedge named X was drawn 50 wide while
|X| is 50 + 15 = 65, and for three sets the child could be larger than the
parent it sat inside (``[1, 2, 3, 4, 5, 6, 7]`` puts ABC=7 inside A=1).  Nothing
in the path checked that a size is a count, or that the triple overlap fits
inside every pair.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plot_api as pa


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_the_two_set_figure_still_labels_the_regions_it_was_given():
    fig = pa.venn([50, 40, 15], set_names=['X', 'Y'])

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert ['50', '40', '15'] == [t for t in texts if t.isdigit()]
    assert 'X' in texts and 'Y' in texts


def test_the_interactive_two_set_sunburst_draws_each_set_at_its_true_total():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.venn([50, 40, 15], set_names=['X', 'Y'], interactive=True)

    trace = fig.data[0]
    assert list(trace.labels) == ['X', 'Y', 'Intersection']
    assert list(trace.values) == [65, 55, 15], "|X| = only-X + both"
    assert list(trace.parents) == ['', '', 'X']


def test_the_interactive_three_set_sunburst_cannot_put_a_child_in_a_smaller_parent():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.venn([10, 12, 8, 4, 3, 2, 1], set_names=['A', 'B', 'C'], interactive=True)

    trace = fig.data[0]
    assert list(trace.values) == [18, 19, 14, 1]
    assert trace.values[0] >= trace.values[3]
    assert trace.values[3] <= min(trace.values[0], trace.values[1], trace.values[2])


@pytest.mark.parametrize('bad', [[50, -40, 15], [-1, 0, 0], [50, 40, -15]])
def test_a_region_cannot_be_a_negative_count(bad):
    with pytest.raises(ValueError, match=r'set_sizes\[\d\] .*>= 0') as info:
        pa.venn(bad)

    assert 'venn()' in str(info.value)


@pytest.mark.parametrize('bad,offender', [
    ([50, 40.5, 15], r'set_sizes\[1\] \(only B\) to be a whole number, got 40\.5'),
    ([2.5, 1, 1], r'set_sizes\[0\] \(only A\) to be a whole number, got 2\.5'),
    (['50', 40, 15], r"set_sizes\[0\] \(only A\) to be a whole number, got '50'"),
    ([1, 2, 3, 4, 5.5, 6, 7], r'set_sizes\[4\] \(AC\) to be a whole number, got 5\.5'),
])
def test_a_region_has_to_be_a_whole_number_and_the_refusal_says_which(bad, offender):
    with pytest.raises(ValueError, match=offender):
        pa.venn(bad)


def test_the_region_that_is_named_in_the_refusal_is_the_one_that_is_wrong():
    with pytest.raises(ValueError, match=r'set_sizes\[2\] \(both\)') as info:
        pa.venn([50, 40, -3])

    assert 'got -3' in str(info.value)


def test_regions_that_add_up_are_still_accepted():
    assert len(pa.venn([0, 0, 0]).axes[0].patches) == 2, "three empty regions is a legal answer"
    assert len(pa.venn([1, 1, 1, 1, 1, 1, 1]).axes[0].patches) == 3


def test_a_triple_overlap_larger_than_a_pair_is_refused():
    with pytest.raises(ValueError, match=r'ABC=7') as info:
        pa.venn([1, 2, 3, 4, 5, 6, 7], set_names=['A', 'B', 'C'])

    text = str(info.value)
    assert 'AB=4, AC=5, BC=6' in text
    assert 'no pair can be smaller than the triple' in text

    with pytest.raises(ValueError, match='ABC=7'):
        pa.venn([1, 2, 3, 4, 5, 6, 7], interactive=True)
