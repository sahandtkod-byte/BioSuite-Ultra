"""``timeseries()`` wants a list of series, and says so when it does not get one.

The loop was ``for i, y in enumerate(ys)`` with no check that ``ys`` holds
series at all.  A caller who handed over one flat list of numbers — the shape of
``y = [1, 2, 3]`` — therefore iterated over *numbers*, and because
``ax.plot(x, 3.0)`` is legal matplotlib, the figure came back with three flat
lines and a legend for them.  A series shorter than ``x`` ended early with no
word said, and extra names were dropped in silence.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

X = [0.0, 1.0, 2.0, 3.0]
SERIES = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_two_series_keep_their_names_and_their_values():
    fig = pa.timeseries(X, SERIES, names=['up', 'down'])

    ax = fig.axes[0]
    assert [line.get_label() for line in ax.lines[:2]] == ['up', 'down']
    assert [np.asarray(line.get_ydata()).tolist() for line in ax.lines[:2]] == SERIES
    assert np.asarray(ax.lines[0].get_xdata()).tolist() == X


def test_a_2_d_array_is_read_as_its_rows():
    fig = pa.timeseries(X, np.asarray(SERIES))

    ax = fig.axes[0]
    assert [line.get_label() for line in ax.lines[:2]] == ['Series 1', 'Series 2']
    assert np.asarray(ax.lines[1].get_ydata()).tolist() == [4.0, 3.0, 2.0, 1.0]


def test_a_flat_sequence_of_numbers_is_refused_as_one_series_wrapped():
    with pytest.raises(ValueError, match=r'flat sequence of 4 plain numbers') as info:
        pa.timeseries(X, [1.0, 2.0, 3.0, 4.0])

    assert 'wrap it in a list' in str(info.value)

    with pytest.raises(ValueError, match='flat sequence of 4 plain numbers'):
        pa.timeseries(X, np.array([1.0, 2.0, 3.0, 4.0]))


def test_a_series_that_ends_before_the_time_axis_is_named_by_index():
    with pytest.raises(ValueError, match=r'4 time points and 3 values in series 1') as info:
        pa.timeseries(X, [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])

    assert 'cannot end before its x axis' in str(info.value)


def test_no_series_at_all_is_refused_in_both_backends():
    with pytest.raises(ValueError, match=r'timeseries\(\) was given no series to draw'):
        pa.timeseries(X, [])

    pytest.importorskip('plotly.graph_objects')
    with pytest.raises(ValueError, match='was given no series to draw'):
        pa.timeseries(X, [], interactive=True)

    with pytest.raises(ValueError, match=r'needs at least one value in ys\[1\]'):
        pa.timeseries(X, [[1.0, 2.0, 3.0, 4.0], []])


def test_names_that_outnumber_the_series_are_refused():
    with pytest.raises(ValueError, match=r'2 series and 3 names') as info:
        pa.timeseries(X, SERIES, names=['a', 'b', 'c'])

    assert 'dropped without the caller hearing' in str(info.value)


def test_fewer_names_than_series_is_allowed_and_says_which_lines_are_auto_named(caplog):
    fig = pa.timeseries(X, SERIES, names=['only'])

    assert [line.get_label() for line in fig.axes[0].lines[:2]] == ['only', 'Series 2']
    records = [r for r in caplog.records if 'take an auto name' in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].name == 'biosuite.plotting.plot_api'
    assert '2 series and 1 names' in records[0].getMessage()
    assert 'at index 1 onward' in records[0].getMessage()
