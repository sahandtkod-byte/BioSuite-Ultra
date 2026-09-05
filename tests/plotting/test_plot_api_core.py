"""Broad contract tests for plotting/plot_api.py (matplotlib path)."""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import plot_api as pa


def _close(fig):
    if fig is not None:
        plt.close(fig)


def test_volcano_matplotlib():
    rng = np.random.default_rng(0)
    log2fc = rng.normal(0, 2, 100)
    pvals = 10 ** rng.uniform(-6, -0.5, 100)
    fig = pa.volcano(log2fc, pvals, gene_names=[f'G{i}' for i in range(100)])
    assert fig is not None
    _close(fig)


def test_pca_matplotlib_df():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(20, 5)),
                      columns=[f'f{i}' for i in range(5)])
    fig = pa.pca(df, labels=[f'S{i}' for i in range(20)])
    assert fig is not None
    _close(fig)


def test_manhattan_matplotlib():
    rng = np.random.default_rng(2)
    chrom = rng.integers(1, 4, 200)
    pos = rng.integers(1, 10_000, 200)
    p = 10 ** rng.uniform(-8, -1, 200)
    fig = pa.manhattan(chrom, pos, p)
    assert fig is not None
    _close(fig)


def test_ma_plot():
    rng = np.random.default_rng(3)
    mean = rng.uniform(100, 10000, 200)
    logfc = rng.normal(0, 1.5, 200)
    fig = pa.ma(mean, logfc)
    assert fig is not None
    _close(fig)


def test_heatmap_and_labels():
    data = np.arange(25).reshape(5, 5) / 10
    fig = pa.heatmap(data, row_labels=[f'r{i}' for i in range(5)],
                     col_labels=[f'c{i}' for i in range(5)])
    assert fig is not None
    _close(fig)


def test_boxplot_dict():
    fig = pa.boxplot({'a': [1, 2, 3, 2.5], 'b': [2, 3, 4, 2.2]})
    assert fig is not None
    _close(fig)


def test_scatter_labels():
    fig = pa.scatter([1, 2, 3], [4, 5, 6], labels=['x', 'y', 'z'])
    assert fig is not None
    _close(fig)


def test_barplot_categories():
    fig = pa.barplot(['a', 'b', 'c'], [3, 7, 2], errors=[0.2, 0.4, 0.2])
    assert fig is not None
    _close(fig)


def test_violin_dict():
    rng = np.random.default_rng(4)
    fig = pa.violin({'grp1': rng.normal(0, 1, 50).tolist(),
                     'grp2': rng.normal(1.5, 1, 50).tolist()})
    assert fig is not None
    _close(fig)


def test_timeseries_named():
    fig = pa.timeseries(np.arange(10), [np.arange(10) * i for i in (1, 2, 3)],
                        names=['s1', 's2', 's3'])
    assert fig is not None
    _close(fig)


def test_qqplot():
    rng = np.random.default_rng(5)
    p = sorted(10 ** rng.uniform(-6, -0.5, 100))
    fig = pa.qqplot(p, interactive=False)
    assert fig is not None
    _close(fig)


def test_venn_two_sets():
    # contract: list of [A, B, AB] sizes
    fig = pa.venn([50, 40, 15], set_names=['X', 'Y'])
    assert fig is not None
    _close(fig)
