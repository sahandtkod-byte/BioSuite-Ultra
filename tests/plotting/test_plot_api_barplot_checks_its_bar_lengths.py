"""Bars, categories and error caps have to count the same thing.

``barplot(['a', 'b', 'c'], [1.0, 2.0])`` died inside matplotlib with
"shape mismatch: objects cannot be broadcast to a single shape. Mismatch is
between arg 0 with shape (3,) and arg 1 with shape (2,)" — which names no
argument the caller passed, and in the Plotly branch a short ``errors`` list was
silently paired with the wrong bars.  A negative uncertainty drew its cap
through the axis, and an empty category list returned a figure with no bars and
no explanation.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

CATEGORIES = ['a', 'b', 'c']
VALUES = [3.0, 7.0, 2.0]
ERRORS = [0.2, 0.4, 0.2]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_three_named_bars_are_still_drawn_at_their_heights():
    fig = pa.barplot(CATEGORIES, VALUES, errors=ERRORS)

    ax = fig.axes[0]
    assert [p.get_height() for p in ax.patches] == [3.0, 7.0, 2.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == CATEGORIES
    assert ax.lines, "the error caps are drawn as lines"


def test_values_that_do_not_match_the_categories_are_refused_by_name():
    with pytest.raises(ValueError, match=r'categories=3, values=2'):
        pa.barplot(CATEGORIES, [1.0, 2.0])

    with pytest.raises(ValueError, match=r'categories=2, values=3'):
        pa.barplot(['a', 'b'], VALUES)


def test_errors_are_counted_against_the_bars_they_cap():
    with pytest.raises(ValueError, match=r'categories=3, errors=2') as info:
        pa.barplot(CATEGORIES, VALUES, errors=[0.1, 0.2])

    assert 'same number of entries' in str(info.value)

    pytest.importorskip('plotly.graph_objects')
    with pytest.raises(ValueError, match=r'categories=3, errors=2'):
        pa.barplot(CATEGORIES, VALUES, errors=[0.1, 0.2], interactive=True)


def test_a_negative_uncertainty_is_refused():
    with pytest.raises(ValueError, match=r'1 negative error value') as info:
        pa.barplot(CATEGORIES, VALUES, errors=[0.1, -0.2, 0.3])

    assert '-0.2' in str(info.value)
    assert 'through the axis' in str(info.value)


def test_no_categories_is_refused_instead_of_drawing_an_empty_axis():
    with pytest.raises(ValueError, match=r'barplot\(\) needs at least one value in categories'):
        pa.barplot([], [])

    pytest.importorskip('plotly.graph_objects')
    with pytest.raises(ValueError, match='needs at least one value in categories'):
        pa.barplot([], [], interactive=True)


def test_one_category_with_one_error_is_a_valid_single_bar():
    with_caps = pa.barplot(['only'], [5.0], errors=[0.5])
    plain = pa.barplot(['only'], [5.0])

    ax = with_caps.axes[0]
    assert [p.get_height() for p in ax.patches] == [5.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ['only']
    assert len(ax.lines) >= 1, "the cap is drawn"
    assert len(plain.axes[0].lines) == 0, "and it is the only thing the errors add"
