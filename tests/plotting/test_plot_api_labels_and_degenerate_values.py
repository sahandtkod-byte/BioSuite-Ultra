"""Labels that arrive as lists, and p-values that arrive as zero.

Each test here pins a shape that used to break ``plot_api`` or ``interactive_plots``
rather than a plot type: sample labels handed over as a Python list (a boolean mask
cannot index one), a p-value of exactly zero (``-log10`` would be infinite), the
colormap name that has to reach Plotly's colourscale, and a colour column whose
groups must come out as one trace each.  The assertion is always about the data the
figure carries, never about the figure existing.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import interactive_plots as ip
from biosuite.plotting import plot_api as pa

HAS_PLOTLY = pa.HAS_PLOTLY
plotly_only = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly missing")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _points(collection):
    return np.asarray(collection.get_offsets())


# ── labels handed over as lists ──────────────────────────────────────────────

@plotly_only
def test_a_pca_grouped_by_a_dataframe_column_carries_the_list_labels_through():
    """The mask indexes the labels, and a plain list cannot be boolean-indexed —
    the labels have to arrive in the hover text of the trace they belong to."""
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 5)),
                         columns=list("abcde"))
    frame["grp"] = ["x"] * 10 + ["y"] * 10
    labels = [f"s{i}" for i in range(20)]

    fig = pa.pca(frame, group_col="grp", labels=labels, interactive=True)

    assert [trace.name for trace in fig.data] == ["x", "y"], [t.name for t in fig.data]
    assert [len(trace.x) for trace in fig.data] == [10, 10]
    assert set(trace.text for trace in fig.data) == {
        tuple(labels[:10]), tuple(labels[10:])}, [t.text for t in fig.data]


def test_a_numeric_dataframe_with_no_grouping_column_draws_one_cloud():
    frame = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 6)))

    fig = pa.pca(frame)

    ax = fig.axes[0]
    assert len(ax.collections) == 1, "nothing was grouped, so nothing is split"
    assert len(_points(ax.collections[0])) == 30
    assert ax.get_xlabel().startswith("PC1 ("), ax.get_xlabel()
    assert ax.get_ylabel().startswith("PC2 ("), ax.get_ylabel()


def test_a_scatter_labels_points_from_a_list_the_caller_supplied():
    fig = pa.scatter([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], labels=["alpha", "beta", "gamma"],
                     show_regression=False)

    ax = fig.axes[0]
    assert np.array_equal(_points(ax.collections[0]),
                          np.column_stack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert len(ax.lines) == 0, "no fit was asked for, so no line is drawn"


# ── p-values at the edge of arithmetic ───────────────────────────────────────

@plotly_only
def test_a_zero_p_value_is_plotted_as_the_floor_the_padding_gives_not_as_infinity():
    fig = pa.volcano([2.0], [0.0], interactive=True)

    heights = np.concatenate([np.asarray(trace.y, dtype=float) for trace in fig.data])
    assert heights.size == 1, "the single gene is drawn once, in one group"
    assert np.all(np.isfinite(heights)), heights
    assert heights[0] == pytest.approx(300.0), "the 1e-300 padding is the height it earns"


@plotly_only
def test_a_qq_plot_of_the_extreme_probabilities_still_has_no_infinite_point():
    fig = pa.qqplot([0.0, 0.5, 1.0], interactive=True)

    y = np.asarray(fig.data[0].y, dtype=float)
    assert np.all(np.isfinite(y)), y
    assert len(y) == 3


def test_a_volcano_split_by_the_cutoffs_reports_the_two_groups_it_used():
    fc = np.array([0.0, 2.5, -3.0, 0.2])
    p = np.array([0.9, 1e-6, 1e-9, 0.5])

    fig = pa.volcano(fc, p, fc_thresh=1.0, p_thresh=0.05)
    ax = fig.axes[0]

    sizes = [len(_points(c)) for c in ax.collections]
    assert sorted(sizes) == [2, 2], "two genes pass both cutoffs, two do not"
    drawn = np.vstack([_points(c) for c in ax.collections])
    assert sorted(drawn[:, 0].tolist()) == pytest.approx(sorted(fc.tolist()))


def test_the_second_chromosome_block_starts_where_the_first_ends_plus_the_gap():
    chroms = ["chr1"] * 3 + ["chr2"] * 3
    pos = np.array([100, 200, 300, 100, 200, 300])

    fig = pa.manhattan(chroms, pos, np.array([0.5] * 6))
    ax = fig.axes[0]
    blocks = [np.asarray(_points(c))[:, 0] for c in ax.collections]

    assert len(blocks) == 2
    assert blocks[0].max() == pytest.approx(300.0)
    assert blocks[1].min() > blocks[0].max(), "the blocks may not overlap"
    assert blocks[1].min() == pytest.approx(300 + 1_000_000 + 100, rel=1e-9), \
        "each chromosome is offset by the previous span plus a megabyte"


# ── the colours and legends the caller asked for ─────────────────────────────

@plotly_only
def test_the_colormap_name_reaches_the_plotly_colourscale():
    fig = pa.heatmap(np.arange(12).reshape(3, 4), cmap="Inferno", interactive=True)

    assert fig.data[0].colorscale[0][1] == "#000004", \
        "Inferno starts at black; Viridis would have meant the argument was ignored"


@plotly_only
def test_a_colour_column_splits_the_points_into_one_trace_per_group():
    fig = ip.interactive_scatter([1, 2, 3, 4], [1, 4, 9, 16],
                                 color_col=["a", "a", "b", "b"])

    assert len(fig.data) == 2, "one trace per group, not one per point"
    assert sorted(len(trace.x) for trace in fig.data) == [2, 2]
    assert {trace.name for trace in fig.data} == {"a", "b"}


@plotly_only
def test_an_interactive_volcano_keeps_a_zero_p_value_finite():
    fig = ip.interactive_volcano([1.0, -2.0], [0.5, 0.0])

    for trace in fig.data:
        assert np.all(np.isfinite(np.asarray(trace.y, dtype=float))), trace.y


def test_a_boxplot_of_two_named_groups_draws_both_boxes():
    fig = pa.boxplot({"t1": [1, 2, 3], "t2": [2, 3, 4]})

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["t1", "t2"]
    assert ax.get_ylabel() == "Value"
    assert ax.get_ylim()[1] >= 4.0
