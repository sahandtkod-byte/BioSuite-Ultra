"""Sample labels have to select the samples they name.

``pca()`` is called with a numpy matrix and a **list** of labels by every one of
the three interfaces — ``gui/tabs/visualization.py``, ``cli/menu.py`` and the
REST endpoint — and the mask that grouping needs was built with ``groups ==
group``.  On a plain list that comparison is the scalar ``False``, and
``coords[False, 0]`` selects nothing, so each group became an empty trace: a
legend with every name in it and not one point drawn, in both backends.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import plot_api as pa

DATA = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 5.0],
                 [4.0, 4.0], [5.0, 1.0], [6.0, 7.0]])
LABELS = ['a', 'a', 'b', 'b', 'c', 'c']


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _counts(ax):
    return [len(np.asarray(collection.get_offsets())) for collection in ax.collections]


def test_a_list_of_sample_labels_draws_every_sample_of_every_group():
    """The regression: six samples in three groups, so 2 + 2 + 2 points."""
    fig = pa.pca(DATA, labels=LABELS)

    ax = fig.axes[0]
    assert _counts(ax) == [2, 2, 2], [t.get_text() for t in ax.get_legend().get_texts()]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['a', 'b', 'c']


def test_a_numpy_label_array_and_a_list_of_the_same_names_draw_the_same_points():
    """The positive control: the two shapes agree, so the test above is counting
    points and not passing on an empty-but-consistent figure."""
    as_list = pa.pca(DATA, labels=LABELS)
    as_array = pa.pca(DATA, labels=np.asarray(LABELS))

    assert (_counts(as_list.axes[0]) == _counts(as_array.axes[0]) == [2, 2, 2])
    first = [np.asarray(c.get_offsets())[:, 0] for c in as_list.axes[0].collections]
    second = [np.asarray(c.get_offsets())[:, 0] for c in as_array.axes[0].collections]
    for a, b in zip(first, second):
        assert np.array_equal(np.sort(a), np.sort(b))


def test_an_interactive_pca_with_list_labels_also_draws_every_sample():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.pca(DATA, labels=LABELS, interactive=True)

    assert [len(trace.x) for trace in fig.data] == [2, 2, 2]
    assert [trace.name for trace in fig.data] == ['a', 'b', 'c']
    assert [list(trace.text) for trace in fig.data] == [['a', 'a'], ['b', 'b'], ['c', 'c']]


def test_labels_that_do_not_cover_every_sample_are_refused():
    with pytest.raises(ValueError, match=r'6 samples and 4 labels'):
        pa.pca(DATA, labels=['a', 'b', 'c', 'd'])

    with pytest.raises(ValueError, match=r'6 samples and 8 labels'):
        pa.pca(DATA, labels=[f's{i}' for i in range(8)])


def test_hover_labels_are_counted_against_a_frame_grouped_by_a_column():
    """The frame path takes its groups from a column, so the labels travel
    separately — and a short list used to die inside numpy when the hover text was
    assembled, naming no argument at all."""
    frame = pd.DataFrame(DATA, columns=['x', 'y'])
    frame['group'] = LABELS

    with pytest.raises(ValueError, match=r'6 samples and 2 labels'):
        pa.pca(frame, group_col='group', labels=['a', 'b'])

    fig = pa.pca(frame, group_col='group', labels=[f's{i}' for i in range(6)],
                 interactive=True)
    assert [len(trace.text) for trace in fig.data] == [2, 2, 2]


def test_grouping_by_a_column_the_frame_does_not_have_is_named():
    frame = pd.DataFrame(DATA, columns=['x', 'y'])
    frame['group'] = LABELS

    with pytest.raises(ValueError, match="to group by 'nope'") as info:
        pa.pca(frame, group_col='nope')

    text = str(info.value)
    assert 'x, y, group' in text, "the columns that do exist are listed"


def test_too_many_components_are_refused_with_the_number_that_can_exist():
    with pytest.raises(ValueError, match=r'4 components from a 6 x 2 matrix.*most.*is 2'):
        pa.pca(DATA, labels=LABELS, n_components=4)


def test_one_component_is_enough_for_a_two_sample_matrix_and_two_is_not():
    fig = pa.pca(np.array([[1.0, 2.0], [3.0, 4.0]]), n_components=1)

    assert _counts(fig.axes[0]) == [2]
    assert fig.axes[0].get_ylabel() == 'PC2', "no second component exists to name"

    with pytest.raises(ValueError, match=r'3 components from a 2 x 2 matrix'):
        pa.pca(np.array([[1.0, 2.0], [3.0, 4.0]]), n_components=3)


def test_a_flat_array_is_refused_before_it_is_mistaken_for_a_matrix():
    with pytest.raises(ValueError, match=r'\(samples x features\) matrix, got a 1-D'):
        pa.pca([1.0, 2.0, 3.0])


def test_a_frame_with_no_numeric_columns_names_the_problem():
    frame = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})

    with pytest.raises(ValueError, match=r'no numeric columns in a frame of 2'):
        pa.pca(frame)


@pytest.mark.parametrize('empty', [np.zeros((0, 3)), pd.DataFrame(np.zeros((0, 2)))])
def test_no_samples_at_all_is_refused_before_sklearn_sees_it(empty):
    with pytest.raises(ValueError, match=r'needs at least one sample row') as info:
        pa.pca(empty)

    assert 'empty matrix' in str(info.value)
