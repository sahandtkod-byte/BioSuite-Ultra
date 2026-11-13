"""A heatmap's grid, and the names hung on it, have to agree.

``heatmap()`` took any array it was handed: a flat list reached ``imshow`` and
raised its complaint about a 1-D image, an empty frame raised inside numpy, and a
``row_labels`` list one entry short silently shifted every remaining name up a
row — the classic way to label the wrong gene.  None of it was the caller's
mistake to read: the function knew the shape it needed.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import plot_api as pa

MATRIX = np.arange(12, dtype=float).reshape(3, 4)
ROWS = ['r1', 'r2', 'r3']
COLS = ['c1', 'c2', 'c3', 'c4']


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_a_matching_grid_and_labels_are_still_drawn():
    fig = pa.heatmap(MATRIX, row_labels=ROWS, col_labels=COLS)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ROWS
    assert [t.get_text() for t in ax.get_xticklabels()] == COLS
    assert np.asarray(ax.images[0].get_array()).shape == (3, 4)


def test_a_flat_array_is_refused_as_not_a_grid():
    with pytest.raises(ValueError, match=r'needs a 2-D grid, got a 1-D array of shape \(4,\)'):
        pa.heatmap([1.0, 2.0, 3.0, 4.0])


def test_a_grid_without_cells_is_refused():
    with pytest.raises(ValueError, match=r'a grid with shape \(0, 3\)') as info:
        pa.heatmap(np.zeros((0, 3)))

    assert 'no cell to colour' in str(info.value)

    with pytest.raises(ValueError, match=r'shape \(3, 0\)'):
        pa.heatmap(np.zeros((3, 0)))


def test_row_labels_are_counted_against_the_rows():
    with pytest.raises(ValueError, match=r'3 rows and 2 row labels'):
        pa.heatmap(MATRIX, row_labels=['a', 'b'])

    with pytest.raises(ValueError, match=r'3 rows and 4 row labels'):
        pa.heatmap(MATRIX, row_labels=['a', 'b', 'c', 'd'])


def test_column_labels_are_counted_against_the_columns():
    with pytest.raises(ValueError, match=r'4 columns and 2 column labels'):
        pa.heatmap(MATRIX, col_labels=['a', 'b'])


def test_a_dataframe_can_still_supply_its_own_names():
    frame = pd.DataFrame(MATRIX, index=ROWS, columns=COLS)

    fig = pa.heatmap(frame)
    ax = fig.axes[0]

    assert [t.get_text() for t in ax.get_yticklabels()] == ROWS
    assert [t.get_text() for t in ax.get_xticklabels()] == COLS

    with pytest.raises(ValueError, match=r'3 rows and 2 row labels'):
        pa.heatmap(frame, row_labels=['nope', 'nada'])


def test_the_interactive_branch_refuses_the_same_mismatches():
    pytest.importorskip('plotly.graph_objects')

    with pytest.raises(ValueError, match=r'4 columns and 3 column'):
        pa.heatmap(MATRIX, col_labels=['a', 'b', 'c'], interactive=True)

    with pytest.raises(ValueError, match=r'needs a 2-D grid'):
        pa.heatmap([1.0, 2.0], interactive=True)
