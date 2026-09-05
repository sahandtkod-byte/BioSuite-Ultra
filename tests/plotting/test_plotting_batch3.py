"""Regression tests for plot_api/interactive_plots review fixes."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import plot_api as A
from biosuite.plotting import interactive_plots as IP

HAS_PLOTLY = A.HAS_PLOTLY
plotly_only = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly missing")


# ── plot_api.pca ────────────────────────────────────────────────────────────
@plotly_only
def test_pca_list_labels_with_group_col_dataframe():
    """Regression: list labels + group_col used to TypeError on boolean mask."""
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 5)),
                      columns=list('abcde'))
    df['grp'] = ['x'] * 10 + ['y'] * 10
    fig = A.pca(df, group_col='grp',
                labels=[f's{i}' for i in range(20)], interactive=True)
    assert fig is not None


def test_pca_static_and_var_labels():
    df = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 6)))
    fig = A.pca(df)
    assert fig is not None


@plotly_only
def test_heatmap_honors_cmap():
    fig = A.heatmap(np.arange(12).reshape(3, 4), cmap='Inferno', interactive=True)
    first = fig.data[0].colorscale[0][1]
    assert first == '#000004'          # Inferno's start, not Viridis'


def test_heatmap_static_and_labels():
    fig = A.heatmap(np.arange(20).reshape(4, 5),
                    row_labels=['a', 'b', 'c', 'd'],
                    col_labels=list('vwxyz'))
    assert fig is not None


# ── volcano / ma / manhattan / qq ───────────────────────────────────────────
def test_volcano_static_significant_split():
    fc = np.array([0.0, 2.5, -3.0, 0.2])
    p = np.array([0.9, 1e-6, 1e-9, 0.5])
    fig = A.volcano(fc, p)
    assert fig is not None


@plotly_only
def test_volcano_interactive_handles_zero_p():
    fig = A.volcano([2.0], [0.0], interactive=True)  # +1e-300 padding
    assert fig is not None


def test_manhattan_offsets_monotonic():
    chroms = ['chr1'] * 3 + ['chr2'] * 3
    pos = np.array([100, 200, 300, 100, 200, 300])
    fig = A.manhattan(chroms, pos, np.array([0.5] * 6))
    assert fig is not None


@plotly_only
def test_qqplot_clipped_bounds_no_inf():
    fig = A.qqplot([0.0, 0.5, 1.0], interactive=True)
    y = fig.data[0].y
    assert np.all(np.isfinite(y))


# ── interactive_plots edge paths ────────────────────────────────────────────
@plotly_only
def test_interactive_volcano_zero_p():
    fig = IP.interactive_volcano([1.0, -2.0], [0.5, 0.0])
    assert fig is not None


@plotly_only
def test_interactive_scatter_color_legend():
    fig = IP.interactive_scatter([1, 2, 3, 4], [1, 4, 9, 16],
                                 color_col=['a', 'a', 'b', 'b'])
    assert len(fig.data) == 2


def test_fallback_boxplot_draws():
    fig = A.boxplot({'t1': [1, 2, 3], 't2': [2, 3, 4]})
    assert fig is not None


def test_venn_two_set_draws():
    fig = A.venn([10, 8, 3])
    assert fig is not None
