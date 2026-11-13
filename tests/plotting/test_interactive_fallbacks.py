"""The matplotlib branch of ``interactive_plots``, with plotly switched off.

Every plot here has two implementations.  ``test_interactive_plots.py`` covers the
plotly one; this file used to claim the fallback without ever disabling plotly,
so it re-tested the plotly branch and the static path stayed unexercised.  The
assertions are about what each fallback draws, not about getting an object back.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip

rng = np.random.default_rng(3)
X = rng.normal(size=20)
Y = rng.normal(size=20)


@pytest.fixture(autouse=True)
def _without_plotly(monkeypatch):
    monkeypatch.setattr(ip, "HAS_PLOTLY", False)


def _first_axes(fig):
    return fig.axes[0]


def _offsets(collection):
    return np.asarray(collection.get_offsets()).tolist()


def test_scatter_fallback_draws_every_point():
    fig = ip.interactive_scatter([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], title="S")
    ax = _first_axes(fig)

    assert _offsets(ax.collections[0]) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert ax.get_title() == "S (static fallback)"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("X", "Y")


def test_scatter_fallback_groups_by_colour_column():
    fig = ip.interactive_scatter([1, 2, 3, 4], [10, 20, 30, 40],
                                 color_col=["a", "b", "a", "b"])
    ax = _first_axes(fig)

    assert len(ax.collections) == 2
    assert {t.get_text() for t in ax.legend().get_texts()} == {"a", "b"}


def test_bar_fallback_uses_the_given_heights_and_categories():
    fig = ip.interactive_bar(["a", "b", "c"], [1.0, 2.5, 3.0], title="B")
    ax = _first_axes(fig)

    assert [p.get_height() for p in ax.containers[0]] == [1.0, 2.5, 3.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert ax.get_title() == "B (static fallback)"


def test_bar_fallback_draws_error_bars():
    from matplotlib.container import BarContainer, ErrorbarContainer

    fig = ip.interactive_bar(["a", "b"], [1.0, 2.0], errors=[0.1, 0.2])
    ax = _first_axes(fig)

    assert [type(c).__name__ for c in ax.containers] == ["BarContainer",
                                                         "ErrorbarContainer"]
    bars, errorbars = ax.containers
    assert isinstance(bars, BarContainer) and isinstance(errorbars, ErrorbarContainer)
    assert [p.get_height() for p in bars] == [1.0, 2.0]
    # Caps sit at value ± error.
    caps = {round(float(v), 3) for line in errorbars.lines[1] for v in line.get_ydata()}
    assert caps == {0.9, 1.1, 1.8, 2.2}


def test_heatmap_fallback_shows_the_matrix_and_a_colorbar():
    data = rng.normal(size=(4, 3))
    fig = ip.interactive_heatmap(data, row_labels=list("wxyz"),
                                 col_labels=list("abc"))
    ax = _first_axes(fig)

    assert len(fig.axes) == 2                     # plot + colourbar
    assert np.array_equal(ax.images[0].get_array(), data)
    assert [t.get_text() for t in ax.get_yticklabels()] == list("wxyz")


def test_volcano_fallback_splits_significant_from_background():
    lfc = np.array([-2.0, -0.5, 0.2, 1.5, 3.0])
    pvals = np.array([0.001, 0.5, 0.9, 0.01, 0.0001])
    fig = ip.interactive_volcano(lfc, pvals)
    ax = _first_axes(fig)

    # |lfc| >= 1 and p < 0.05 -> points 0, 3 and 4 are the significant ones.
    assert len(ax.collections[0].get_offsets()) == 2
    assert len(ax.collections[1].get_offsets()) == 3
    assert ax.get_ylabel() == "-log10(p-value)"


def test_line_fallback_plots_every_series_with_its_name():
    fig = ip.interactive_line([0, 1, 2], [[1, 2, 3], [3, 2, 1]], names=["up", "down"])
    ax = _first_axes(fig)

    assert [line.get_ydata().tolist() for line in ax.lines] == [[1, 2, 3], [3, 2, 1]]
    assert {t.get_text() for t in ax.legend().get_texts()} == {"up", "down"}


def test_3d_scatter_has_no_fallback():
    assert ip.interactive_3d_scatter(X, Y, X) is None


def test_pie_fallback_draws_one_wedge_per_label():
    fig = ip.interactive_pie(["A", "B", "C"], [3.0, 2.0, 5.0])
    ax = _first_axes(fig)

    assert len(ax.patches) == 3
    assert {"A", "B", "C"} <= {t.get_text() for t in ax.texts}


def test_boxplot_fallback_labels_the_groups_and_spans_their_data():
    data = {"ctrl": X, "treat": Y + 5}
    fig = ip.interactive_boxplot(data, y_label="Signal")
    ax = _first_axes(fig)

    assert [t.get_text() for t in ax.get_xticklabels()] == ["ctrl", "treat"]
    assert ax.get_ylabel() == "Signal"
    # Both groups reached the axes: the y range covers the union of their values.
    assert ax.get_ylim()[0] <= min(X.min(), (Y + 5).min())
    assert ax.get_ylim()[1] >= max(X.max(), (Y + 5).max())


def test_violin_fallback_draws_one_violin_per_group():
    data = {"a": X, "b": Y, "c": X + 1}
    fig = ip.interactive_violin(data, y_label="Signal")
    ax = _first_axes(fig)

    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert ax.get_ylabel() == "Signal"
    values = np.concatenate(list(data.values()))
    assert ax.get_ylim()[0] <= values.min() and ax.get_ylim()[1] >= values.max()


def test_pca_fallback_uses_variance_explained_in_the_axis_labels():
    coords = np.column_stack([X, Y])
    fig = ip.interactive_pca(coords, variance_explained=[0.4, 0.25])
    ax = _first_axes(fig)

    assert np.allclose(np.asarray(_offsets(ax.collections[0])), coords)
    assert ax.get_xlabel() == "PC1 (40.0%)"
    assert ax.get_ylabel() == "PC2 (25.0%)"


def test_pca_fallback_can_split_the_points_by_a_colour_column():
    coords = np.column_stack([X, Y])
    colours = ["tumor"] * 10 + ["normal"] * 10

    fig = ip.interactive_pca(coords, color_col=colours)

    ax = _first_axes(fig)
    assert len(ax.collections) == 2, "one scatter per group"
    assert sorted(t.get_text() for t in ax.legend().get_texts()) == ["normal", "tumor"]
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(coords)


def test_heatmap_fallback_without_labels_leaves_the_axes_as_imshow_set_it():
    fig = ip.interactive_heatmap([[1.0, 2.0], [3.0, 4.0]])
    ax = _first_axes(fig)

    assert ax.get_xlim() == pytest.approx((-0.5, 1.5)), "two columns, numbered by matplotlib"
    assert ax.get_ylim() == pytest.approx((1.5, -0.5)), "imshow reads the first row from the top"
    assert len(fig.axes) == 2, "the colourbar still came with it"


# ── the module as it loads when plotly is not installed ──────────────────────

def test_the_module_loads_without_plotly_and_keeps_every_fallback(monkeypatch):
    """``HAS_PLOTLY`` is decided once, at import; re-importing the module shows
    that the whole module — not just one patched flag — degrades to matplotlib."""
    import importlib.util
    import pathlib
    import sys

    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.interactive_plots_without_plotly",
        pathlib.Path(ip.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "plotly", None)
    spec.loader.exec_module(module)

    assert module.HAS_PLOTLY is False

    fig = module.interactive_scatter(X, Y)

    assert isinstance(fig, plt.Figure)
    assert len(_first_axes(fig).collections) == 1


def test_manhattan_fallback_plots_one_collection_per_chromosome():
    chroms = ["1", "1", "2", "2", "2"]
    positions = [100, 200, 50, 60, 70]
    neg_log_p = [3.0, 4.0, 2.0, 6.0, 1.0]
    fig = ip.interactive_manhattan(chroms, positions, neg_log_p)
    ax = _first_axes(fig)

    assert len(ax.collections) == 2
    plotted = sum(len(c.get_offsets()) for c in ax.collections)
    assert plotted == len(positions)


def test_ma_fallback_marks_the_significant_points_separately():
    mean = np.array([1.0, 2.0, 3.0, 4.0])
    log_fc = np.array([0.1, 1.5, -0.2, 2.0])
    fig = ip.interactive_ma(mean, log_fc)
    ax = _first_axes(fig)

    assert len(ax.collections[0].get_offsets()) == 2       # |log2FC| < 1
    assert len(ax.collections[1].get_offsets()) == 2       # |log2FC| >= 1
    assert [line.get_ydata()[0] for line in ax.lines] == [0.0]


def test_qq_fallback_draws_points_and_the_reference_line():
    pvals = np.clip(rng.random(12), 1e-6, 1 - 1e-6)
    fig = ip.interactive_qq(pvals)
    ax = _first_axes(fig)

    assert len(ax.collections[0].get_offsets()) == len(pvals)
    assert len(ax.lines) == 1


def test_dotplot_fallback_labels_genes_on_the_y_axis():
    fig = ip.interactive_dotplot(["c1", "c2", "c1", "c2"], ["g1", "g2", "g1", "g2"],
                                 [0.1, 0.2, 0.3, 0.4], sizes=[10, 20, 30, 40])
    ax = _first_axes(fig)

    assert _offsets(ax.collections[0]) == [[0, 0], [1, 1], [0, 2], [1, 3]]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["g1", "g2", "g1", "g2"]
    assert len(fig.axes) == 2                              # colourbar


def test_upset_sunburst_fallback_falls_back_to_a_pie_of_the_sets():
    fig = ip.interactive_upset_sunburst({"A": 10, "B": 5}, {"A&B": 2})
    ax = _first_axes(fig)

    assert len(ax.patches) == 2
    assert {"A", "B"} <= {t.get_text() for t in ax.texts}


def test_export_interactive_report_refuses_without_plotly(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert ip.export_interactive_report({"a": object()}, str(out)) is None

    assert "Plotly not installed" in capsys.readouterr().out
    assert not out.exists()
