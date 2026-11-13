"""``interactive_pca`` takes the table a PCA function returns, in either backend.

The coordinate table was indexed as ``coords[i, 0]``, which is a numpy idea: a
caller handing over the list-of-lists that any PCA routine naturally returns got
``TypeError: list indices must be integers or slices, not tuple`` — from the
plotly branch before it drew anything, and from the fallback after it had opened
a figure.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip

TABLE = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_list_of_samples_is_a_coordinate_table():
    fig = ip.interactive_pca(TABLE)

    assert list(fig.data[0].x) == [0.0, 2.0, 4.0]
    assert list(fig.data[0].y) == [1.0, 3.0, 5.0]


def test_the_same_table_works_in_the_fallback():
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_pca(TABLE)
    finally:
        ip.HAS_PLOTLY = True

    offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
    assert offsets.tolist() == TABLE


def test_the_fallback_splits_a_list_table_by_colour_column():
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_pca(TABLE, color_col=["a", "a", "b"])
    finally:
        ip.HAS_PLOTLY = True

    assert len(fig.axes[0].collections) == 2


@pytest.mark.parametrize("coords, reason", [
    ([1.0, 2.0, 3.0], "needs a 2-D table"),
    (np.array([[1.0], [2.0]]), "at least two components"),
    (np.zeros((0, 2)), "needs at least one value"),
])
def test_a_table_that_is_not_a_pca_result_is_refused_by_name(coords, reason):
    with pytest.raises(ValueError) as excinfo:
        ip.interactive_pca(coords)

    assert "interactive_pca()" in str(excinfo.value)
    assert reason in str(excinfo.value), str(excinfo.value)


def test_labels_must_be_one_per_sample():
    with pytest.raises(ValueError, match="coords=3, labels=2"):
        ip.interactive_pca(TABLE, labels=["s1", "s2"])


def test_the_variance_goes_into_the_axis_titles():
    fig = ip.interactive_pca(TABLE, variance_explained=[0.42, 0.17, 0.05])

    assert fig.layout.xaxis.title.text == "PC1 (42.0%)"
    assert fig.layout.yaxis.title.text == "PC2 (17.0%)"


def test_a_single_variance_is_not_enough_to_label_an_axis():
    """The positive control for the guard above: one number names no pair of axes."""
    fig = ip.interactive_pca(TABLE, variance_explained=[0.42])

    assert fig.layout.xaxis.title.text == "PC1"
    assert fig.layout.yaxis.title.text == "PC2"
