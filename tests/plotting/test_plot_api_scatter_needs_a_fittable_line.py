"""A regression line needs data that can have one.

``scatter(show_regression=True)`` fitted whatever it was handed and printed the
result as a measurement: one point made scipy answer with NaN, and the legend
read ``R²=nan, p=nan``; a NaN anywhere in x or y did the same without even
warning; identical x values raised scipy's own sentence about an infinite slope;
and no points at all died inside ``np.min`` on an empty array.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

X = [0.0, 1.0, 2.0]
Y = [1.0, 3.0, 5.0]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_a_real_fit_draws_the_line_it_describes_and_prints_no_nan():
    fig = pa.scatter(X, Y)

    ax = fig.axes[0]
    line = ax.lines[0]
    xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    assert xs.size == 100
    assert (xs[0], xs[-1]) == (0.0, 2.0)
    assert (ys[0], ys[-1]) == pytest.approx((1.0, 5.0)), "y = 2x + 1, the fit these points lie on"
    label = ax.get_legend().get_texts()[0].get_text()
    assert label.startswith('R²=1.000'), label
    assert 'nan' not in label.lower(), label


def test_a_single_point_refuses_the_fit_and_tells_the_caller_the_way_out():
    with pytest.raises(ValueError, match=r'cannot fit a line to 1 point.*show_regression=False'):
        pa.scatter([1.0], [2.0])

    fig = pa.scatter([1.0], [2.0], show_regression=False)
    assert len(np.asarray(fig.axes[0].collections[0].get_offsets())) == 1
    assert len(fig.axes[0].lines) == 0, "no fit was possible, so no line is claimed"


def test_identical_x_values_are_refused_instead_of_printing_an_infinite_slope():
    with pytest.raises(ValueError, match=r'every x is 2\.0') as info:
        pa.scatter([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

    assert 'slope is infinite' in str(info.value)


def test_a_non_finite_point_is_named_by_index_rather_than_dropped_silently():
    with pytest.raises(ValueError, match=r'1 of 3 points that are not finite') as info:
        pa.scatter([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])

    assert 'first at index 2' in str(info.value)

    with pytest.raises(ValueError, match=r'2 of 4 points that are not finite'):
        pa.scatter([1.0, 2.0, 3.0, 4.0], [1.0, np.inf, 3.0, np.nan])


def test_labels_have_to_cover_every_point():
    with pytest.raises(ValueError, match=r'4 points and 3 labels'):
        pa.scatter([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], labels=['a', 'b', 'c'])


def test_no_points_is_refused_in_both_backends():
    with pytest.raises(ValueError, match=r'scatter\(\) needs at least one value in x'):
        pa.scatter([], [])

    with pytest.raises(ValueError, match=r'x=2, y=3'):
        pa.scatter([1.0, 2.0], [1.0, 2.0, 3.0])


def test_the_interactive_branch_refuses_the_same_fits_as_the_static_one():
    """The fit is computed before either backend runs, so Plotly JSON cannot be
    handed back with ``R²=nan`` in a trace name."""
    pytest.importorskip('plotly.graph_objects')

    with pytest.raises(ValueError, match=r'cannot fit a line to 1 point'):
        pa.scatter([1.0], [2.0], interactive=True)

    fig = pa.scatter(X, Y, interactive=True)
    assert len(fig.data) == 2
    assert fig.data[1].name == 'R²=1.000'
    assert np.asarray(fig.data[1].y, dtype=float)[0] == pytest.approx(1.0)
