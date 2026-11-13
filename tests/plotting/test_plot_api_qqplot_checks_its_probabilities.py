"""A Q-Q plot of probabilities that are not probabilities.

``qqplot()`` clipped its input into ``[1e-16, 1 - 1e-16]`` before taking the
normal quantile, and the clip ran on data that was already outside [0, 1]: a
p-value of ``-0.5`` arrived at the same height as ``p = 1e-16``, and ``1.4`` at
the same height as the largest possible p — measured as ``-8.222`` and ``+8.210``
on the plot.  The impossible value was therefore drawn as the most extreme point
on the chart, which is the opposite of a warning.  An empty input died inside
``min()`` with numpy's own complaint.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _points(fig):
    return np.asarray(fig.axes[0].collections[0].get_offsets())


def test_the_extremes_of_the_legal_range_stay_finite_and_are_pinned_to_their_quantiles():
    fig = pa.qqplot([0.0, 0.5, 1.0])

    drawn = _points(fig)
    assert drawn.shape == (3, 2)
    assert np.all(np.isfinite(drawn))
    assert drawn[:, 0].tolist() == pytest.approx([-0.6744897501960817, 0.0, 0.6744897501960817])
    assert drawn[:, 1].tolist() == pytest.approx([-8.222082216130435, 0.0, 8.209536151601387])


def test_probabilities_outside_the_unit_interval_are_counted_and_named():
    with pytest.raises(ValueError, match=r'2 of 3 pvalues outside \[0, 1\]') as info:
        pa.qqplot([1.4, 0.5, -0.5])

    text = str(info.value)
    assert '-0.5' in text and '1.4' in text
    assert 'most extreme point' in text, "the reason the clip was wrong, in the caller's words"


def test_a_single_outlier_is_enough_to_be_refused():
    with pytest.raises(ValueError, match=r'1 of 4 pvalues outside'):
        pa.qqplot([0.1, 0.2, 0.3, 1.0001])


def test_no_pvalues_is_refused_before_the_identity_line_is_computed():
    with pytest.raises(ValueError, match=r'qqplot\(\) needs at least one value in pvalues'):
        pa.qqplot([])

    pytest.importorskip('plotly.graph_objects')
    with pytest.raises(ValueError, match='needs at least one value in pvalues'):
        pa.qqplot([], interactive=True)


def test_a_column_vector_is_refused_as_not_a_list_of_pvalues():
    with pytest.raises(ValueError, match='must be a 1-D sequence'):
        pa.qqplot(np.array([[0.1], [0.2], [0.3]]))


def test_the_interactive_branch_refuses_the_same_impossible_values():
    pytest.importorskip('plotly.graph_objects')

    with pytest.raises(ValueError, match=r'outside \[0, 1\]'):
        pa.qqplot([0.5, 2.0], interactive=True)

    fig = pa.qqplot([0.01, 0.5, 0.99], interactive=True)
    assert np.all(np.isfinite(np.asarray(fig.data[0].y, dtype=float)))
