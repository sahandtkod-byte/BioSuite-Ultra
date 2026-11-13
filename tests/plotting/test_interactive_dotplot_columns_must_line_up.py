"""A dot plot has four columns, and they have to describe the same dots.

The plotly branch zips categories, genes, values and sizes for the hover text,
so a caller who passed three categories, two genes and one value used to get a
trace with three x values, two y values and two hover strings — missing cells,
with nothing said.  The matplotlib fallback also had no x tick labels at all, so
a saved dot plot showed its cell types as 0, 1, 2.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("columns", [
    dict(categories=["c1", "c2", "c3"], genes=["g1", "g2"], values=[1.0]),
    dict(categories=["c1", "c2"], genes=["g1", "g2"], values=[1.0, 2.0],
         sizes=[5]),
])
def test_columns_that_do_not_line_up_are_refused(columns):
    with pytest.raises(ValueError) as excinfo:
        ip.interactive_dotplot(**columns)

    assert "every column of a plot must describe the same number of entries" \
        in str(excinfo.value), str(excinfo.value)


def test_the_trace_holds_one_dot_per_column_entry():
    fig = ip.interactive_dotplot(["c1", "c2"], ["g1", "g2"], [0.1, 0.2],
                                 sizes=[10, 20])

    trace = fig.data[0]
    assert len(trace.x) == len(trace.y) == len(trace.text) == 2
    assert trace.text[0] == "c1: g1<br>Expr: 0.10<br>%: 10%"


def test_a_size_is_reported_as_the_percentage_the_hover_calls_it():
    fig = ip.interactive_dotplot(["c1"], ["g1"], [0.5], sizes=[37])

    assert "%: 37%" in fig.data[0].text[0], fig.data[0].text[0]


def test_without_sizes_every_dot_is_uniform():
    fig = ip.interactive_dotplot(["c1", "c2"], ["g1", "g2"], [0.1, 0.2])

    assert list(fig.data[0].marker.size) == [10, 10]


def test_the_fallback_names_its_categories_on_the_x_axis():
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_dotplot(["c1", "c2", "c1", "c2"],
                                     ["g1", "g2", "g1", "g2"],
                                     [0.1, 0.2, 0.3, 0.4], sizes=[10, 20, 30, 40])
    finally:
        ip.HAS_PLOTLY = True

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c1", "c2"]
    assert np.asarray(ax.collections[0].get_offsets()).tolist() == \
        [[0, 0], [1, 1], [0, 2], [1, 3]], "one column per category, as in plotly"


def test_the_fallback_keeps_numeric_categories_on_their_own_scale():
    """The tick labels are a string convenience; a numeric x axis must stay numeric."""
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_dotplot([5, 10, 5], ["g1", "g2", "g3"], [1.0, 2.0, 3.0])
    finally:
        ip.HAS_PLOTLY = True

    xs = np.asarray(fig.axes[0].collections[0].get_offsets())[:, 0]
    assert sorted(set(xs.tolist())) == [5.0, 10.0]
