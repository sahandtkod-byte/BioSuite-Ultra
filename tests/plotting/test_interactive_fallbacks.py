"""Interactive plots minus plotly: matplotlib fallback contracts."""
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip


rng = np.random.default_rng(3)
X = rng.normal(size=20)
Y = rng.normal(size=20)


def test_scatter_contract():
    out = ip.interactive_scatter(X, Y, title='T')
    assert out is not None


def test_bar_and_heatmap():
    assert ip.interactive_bar(['a', 'b', 'c'], [1.0, 2.0, 3.0]) is not None
    assert ip.interactive_heatmap(rng.normal(size=(4, 3))) is not None


def test_volcano_and_line():
    lfc = rng.normal(size=40)
    p = -np.log10(np.clip(rng.random(40), 1e-9, 1))
    assert ip.interactive_volcano(lfc, p) is not None
    assert ip.interactive_line(X, [Y, Y * 1.1], names=['a', 'b']) is not None


def test_3d_and_pie():
    assert ip.interactive_3d_scatter(X, Y, X) is not None
    assert ip.interactive_pie(['A', 'B'], [3.0, 7.0]) is not None


def test_boxplot_and_violin():
    assert ip.interactive_boxplot({'a': X, 'b': Y}) is not None
    assert ip.interactive_violin({'a': X, 'b': Y}) is not None


def test_pca_manhattan_ma_qq():
    coords = rng.normal(size=(30, 2))
    assert ip.interactive_pca(coords) is not None
    assert ip.interactive_manhattan(['1'] * 30, np.arange(30) * 100, rng.random(30) * 5) is not None
    assert ip.interactive_ma(rng.normal(8, 1, 30), rng.normal(size=30)) is not None
    assert ip.interactive_qq(np.clip(rng.random(30), 1e-6, 0.99)) is not None


def test_upset_sunburst_and_dotplot():
    assert ip.interactive_upset_sunburst({'A': 10, 'B': 5}, {'A&B': 2}) is not None
    n = 4
    assert ip.interactive_dotplot(['t1'] * n, ['g1', 'g2', 'g1', 'g2'],
                                  rng.random(n), sizes=10 + 10 * rng.random(n)) is not None


def test_export_report(tmp_path):
    plot_dict = {
        'scatter': ip.interactive_scatter(X, Y),
        'bar': ip.interactive_bar(['x'], [1.0]),
    }
    out = tmp_path / 'rep.html'
    ip.export_interactive_report(plot_dict, str(out))
    assert out.exists()
