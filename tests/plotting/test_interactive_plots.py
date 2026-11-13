"""The plotly branch of ``interactive_plots``: each function must carry its data.

``assert fig is not None`` was the whole content of most of these tests, which
proved nothing: a trace with the wrong numbers, the wrong name or no hover text
still passed.  plotly is a declared dependency, so the traces are inspected
directly and the expected values are worked out from the documented thresholds
rather than copied from whatever the code happened to return.  The matplotlib
fallbacks are the subject of ``test_interactive_fallbacks.py``, which forces that
branch on.
"""
import numpy as np
import plotly.graph_objects as go
import pytest
from scipy import stats

from biosuite.plotting.interactive_plots import (
    export_interactive_report,
    interactive_3d_scatter,
    interactive_bar,
    interactive_boxplot,
    interactive_dotplot,
    interactive_heatmap,
    interactive_line,
    interactive_ma,
    interactive_manhattan,
    interactive_pca,
    interactive_pie,
    interactive_qq,
    interactive_scatter,
    interactive_upset_sunburst,
    interactive_violin,
    interactive_volcano,
)

rng = np.random.default_rng(11)


def points(trace):
    return list(zip(list(trace.x), list(trace.y)))


class TestScatter:
    def test_carries_the_coordinates_and_title(self):
        fig = interactive_scatter([1.0, 2.0, 3.0, 4.0],
                                  [10.0, 20.0, 30.0, 40.0], title="Test Scatter")

        assert isinstance(fig, go.Figure)
        assert points(fig.data[0]) == [(1.0, 10.0), (2.0, 20.0),
                                       (3.0, 30.0), (4.0, 40.0)]
        assert fig.layout.title.text == "Test Scatter"
        assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("X", "Y")

    def test_hover_text_uses_the_labels(self):
        fig = interactive_scatter([1, 2, 3], [4, 5, 6], labels=["A", "B", "C"])

        assert list(fig.data[0].text) == ["A", "B", "C"]
        assert fig.data[0].hoverinfo == "text"

    def test_without_labels_the_hover_falls_back_to_coordinates(self):
        fig = interactive_scatter([1.5, 2.25], [3.5, 4.75])

        assert list(fig.data[0].text) == ["x=1.50, y=3.50", "x=2.25, y=4.75"]

    def test_colour_column_creates_one_named_trace_per_class(self):
        fig = interactive_scatter([1, 2, 3, 4], [10, 20, 30, 40],
                                  color_col=["cat1", "cat1", "cat2", "cat2"])

        by_name = {trace.name: trace for trace in fig.data}
        assert set(by_name) == {"cat1", "cat2"}
        assert points(by_name["cat1"]) == [(1, 10), (2, 20)]
        assert points(by_name["cat2"]) == [(3, 30), (4, 40)]

    def test_colour_column_and_labels_together(self):
        fig = interactive_scatter([1, 2, 3, 4], [10, 20, 30, 40],
                                  labels=["a", "b", "c", "d"],
                                  color_col=["X", "X", "Y", "Y"])

        by_name = {trace.name: trace for trace in fig.data}
        assert sorted(by_name["X"].text) == ["a", "b"]
        assert sorted(by_name["Y"].text) == ["c", "d"]

    def test_numpy_arrays_are_plotted_as_is(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig = interactive_scatter(x, y)

        assert len(fig.data[0].x) == 50
        assert np.allclose(np.asarray(fig.data[0].y, dtype=float), y)

    def test_single_point(self):
        assert points(interactive_scatter([5.0], [10.0]).data[0]) == [(5.0, 10.0)]

    def test_output_html_is_written_and_interactive(self, tmp_path):
        out = tmp_path / "scatter.html"
        interactive_scatter([1, 2, 3], [4, 5, 6], output_html=str(out))

        assert out.stat().st_size > 1000
        assert "plotly" in out.read_text(encoding="utf-8").lower()


class TestBar:
    def test_categories_and_values_become_one_bar_trace(self):
        fig = interactive_bar(["a", "b", "c"], [1.0, 2.0, 3.0], title="B")

        assert list(fig.data[0].x) == ["a", "b", "c"]
        assert list(fig.data[0].y) == [1.0, 2.0, 3.0]
        assert fig.data[0].error_y.array is None
        assert fig.layout.title.text == "B"

    def test_errors_become_error_bars(self):
        fig = interactive_bar(["a", "b"], [1.0, 2.0], errors=[0.1, 0.2])

        assert fig.data[0].error_y.type == "data"
        assert list(fig.data[0].error_y.array) == [0.1, 0.2]

    def test_thirty_categories_all_reach_the_trace(self):
        labels = [f"c{i}" for i in range(30)]
        fig = interactive_bar(labels, list(range(30)))

        assert list(fig.data[0].x) == labels
        assert list(fig.data[0].y)[-1] == 29

    def test_single_category(self):
        assert points(interactive_bar(["only"], [7.0]).data[0]) == [("only", 7.0)]

    def test_output_html(self, tmp_path):
        out = tmp_path / "bar.html"
        interactive_bar(["a", "b"], [1, 2], output_html=str(out))

        assert out.stat().st_size > 1000


class TestHeatmap:
    def test_matrix_is_the_trace_z(self):
        data = rng.normal(size=(4, 3))
        fig = interactive_heatmap(data)

        assert np.allclose(np.asarray(fig.data[0].z), data)
        # plotly expands a named scale into its stops; Viridis starts at this one.
        assert fig.data[0].colorscale[0][1] == "#440154"

    def test_labels_land_on_their_axes(self):
        fig = interactive_heatmap(np.ones((2, 2)), row_labels=list("xy"),
                                  col_labels=list("ab"))

        assert list(fig.data[0].y) == ["x", "y"]
        assert list(fig.data[0].x) == ["a", "b"]

    def test_custom_colorscale_reaches_the_trace(self):
        scale = [[0.0, "red"], [1.0, "blue"]]
        fig = interactive_heatmap(np.ones((2, 2)), colorscale=scale)

        assert [list(stop) for stop in fig.data[0].colorscale] == scale

    def test_hover_template_reports_the_value(self):
        assert "%{z" in interactive_heatmap(np.ones((2, 2))).data[0].hovertemplate

    def test_1x1_matrix(self):
        z = interactive_heatmap(np.array([[5.0]])).data[0].z

        assert np.asarray(z).shape == (1, 1)

    def test_output_html(self, tmp_path):
        out = tmp_path / "heat.html"
        interactive_heatmap(rng.normal(size=(3, 3)), output_html=str(out))

        assert out.stat().st_size > 1000


class TestVolcano:
    LFC = np.array([-2.0, -0.5, 0.2, 1.5, 3.0])
    PVALS = np.array([0.001, 0.5, 0.9, 0.01, 0.0001])
    # Significant means |log2FC| >= 1.0 and p < 0.05: samples 0, 3 and 4, whose
    # -log10(p) are 3.0, 2.0 and 4.0.
    SIGNIFICANT = [(-2.0, 3.0), (1.5, 2.0), (3.0, 4.0)]

    def test_splits_significant_from_background(self):
        fig = interactive_volcano(self.LFC, self.PVALS)

        not_sig, sig = fig.data
        assert (not_sig.name, sig.name) == ("Not significant", "Significant")
        assert [(float(x), round(float(y), 3)) for x, y in points(sig)] == self.SIGNIFICANT
        assert [float(x) for x in not_sig.x] == [-0.5, 0.2]

    def test_hover_names_only_the_significant_genes(self):
        genes = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]
        fig = interactive_volcano(self.LFC, self.PVALS, gene_names=genes)
        hover = " ".join(str(text) for text in fig.data[1].text)

        for gene in ("TP53", "MYC", "KRAS"):
            assert gene in hover
        for dropped in ("BRCA1", "EGFR"):
            assert dropped not in hover

    def test_loose_thresholds_leave_nothing_in_background(self):
        fig = interactive_volcano(self.LFC, self.PVALS, fc_thresh=0.1, p_thresh=0.99)

        assert len(fig.data[1].x) == 5
        assert len(fig.data[0].x) == 0

    def test_strict_thresholds_leave_nothing_significant(self):
        fig = interactive_volcano([0.1, 0.2], [0.5, 0.9])

        assert len(fig.data[0].x) == 2
        assert len(fig.data[1].x) == 0

    def test_axis_titles(self):
        fig = interactive_volcano(self.LFC, self.PVALS)

        assert fig.layout.xaxis.title.text == "Log2 Fold Change"
        assert fig.layout.yaxis.title.text == "-log10(p-value)"

    def test_output_html(self, tmp_path):
        out = tmp_path / "volcano.html"
        interactive_volcano(self.LFC, self.PVALS, output_html=str(out))

        assert out.stat().st_size > 1000


class TestLine:
    def test_one_trace_per_series(self):
        fig = interactive_line([0, 1, 2], [[1, 2, 3], [3, 2, 1]])

        assert [t.name for t in fig.data] == ["Series 1", "Series 2"]
        assert list(fig.data[1].y) == [3, 2, 1]

    def test_names_are_used(self):
        fig = interactive_line([0, 1], [[1, 2], [3, 4]], names=["up", "down"])

        assert [t.name for t in fig.data] == ["up", "down"]

    def test_single_line(self):
        assert points(interactive_line([1, 2, 3], [[4, 5, 6]]).data[0]) \
            == [(1, 4), (2, 5), (3, 6)]

    def test_twelve_series(self):
        fig = interactive_line(list(range(5)), [list(range(5)) for _ in range(12)])

        assert len(fig.data) == 12

    def test_output_html(self, tmp_path):
        out = tmp_path / "line.html"
        interactive_line([1, 2, 3], [[4, 5, 6]], output_html=str(out))

        assert out.stat().st_size > 1000


class Test3DScatter:
    def test_coordinates_reach_the_trace(self):
        fig = interactive_3d_scatter([1, 2], [3, 4], [5, 6], title="3D")

        assert isinstance(fig.data[0], go.Scatter3d)
        assert (list(fig.data[0].x), list(fig.data[0].y), list(fig.data[0].z)) \
            == ([1, 2], [3, 4], [5, 6])
        assert fig.layout.title.text == "3D"

    def test_labels_become_hover_text(self):
        fig = interactive_3d_scatter([1, 2], [3, 4], [5, 6], labels=["p", "q"])

        assert list(fig.data[0].text) == ["p", "q"]

    def test_colour_column_splits_into_traces(self):
        fig = interactive_3d_scatter([1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8],
                                     color_col=["g1", "g1", "g2", "g2"])

        assert {t.name for t in fig.data} == {"g1", "g2"}
        assert all(isinstance(t, go.Scatter3d) for t in fig.data)

    def test_numpy_arrays(self):
        fig = interactive_3d_scatter(np.arange(10.0), np.arange(10.0) ** 2,
                                     np.arange(10.0) - 5)

        assert len(fig.data[0].z) == 10

    def test_output_html(self, tmp_path):
        out = tmp_path / "three_d.html"
        interactive_3d_scatter([1, 2], [3, 4], [5, 6], output_html=str(out))

        assert out.stat().st_size > 1000


class TestPie:
    def test_labels_and_values(self):
        fig = interactive_pie(["a", "b", "c"], [1, 2, 3])

        assert list(fig.data[0].labels) == ["a", "b", "c"]
        assert list(fig.data[0].values) == [1, 2, 3]
        assert fig.data[0].hole == 0.3

    def test_nine_slices(self):
        fig = interactive_pie([f"s{i}" for i in range(9)], list(range(1, 10)))

        assert len(fig.data[0].labels) == 9

    def test_single_slice(self):
        assert list(interactive_pie(["only"], [1]).data[0].labels) == ["only"]

    def test_output_html(self, tmp_path):
        out = tmp_path / "pie.html"
        interactive_pie(["a", "b"], [1, 2], output_html=str(out))

        assert out.stat().st_size > 1000


class TestBoxplotAndViolin:
    def test_boxplot_one_trace_per_group(self):
        data = {"ctrl": list(range(6)), "treat": [2, 3, 4, 5, 6, 7]}
        fig = interactive_boxplot(data, title="BX")

        assert [t.name for t in fig.data] == ["ctrl", "treat"]
        assert list(fig.data[0].y) == list(range(6))
        assert fig.data[0].boxmean == "sd"
        assert fig.layout.title.text == "BX"

    def test_boxplot_single_group(self):
        fig = interactive_boxplot({"only": [1.0, 2.0, 3.0]})

        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Box)

    def test_boxplot_eight_groups(self):
        fig = interactive_boxplot({f"g{i}": [i, i + 1] for i in range(8)})

        assert len(fig.data) == 8

    def test_boxplot_output_html(self, tmp_path):
        out = tmp_path / "box.html"
        interactive_boxplot({"a": [1, 2]}, output_html=str(out))

        assert out.stat().st_size > 1000

    def test_violin_one_trace_per_group_with_mean_line(self):
        fig = interactive_violin({"a": [1, 2, 3], "b": [2, 3, 4]})

        assert [t.name for t in fig.data] == ["a", "b"]
        assert all(isinstance(t, go.Violin) for t in fig.data)
        assert fig.data[0].meanline.visible is True
        assert fig.data[0].box.visible is True


class TestPCAAndGenomeWide:
    def test_pca_plots_the_first_two_components(self):
        coords = np.column_stack([np.arange(6.0), np.arange(6.0) * -1, np.zeros(6)])
        fig = interactive_pca(coords)

        assert points(fig.data[0]) == [(float(i), float(-i)) for i in range(6)]
        assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("PC1", "PC2")

    def test_pca_variance_explained_goes_into_the_axis_labels(self):
        fig = interactive_pca(np.zeros((4, 2)), variance_explained=[0.42, 0.18])

        assert fig.layout.xaxis.title.text == "PC1 (42.0%)"
        assert fig.layout.yaxis.title.text == "PC2 (18.0%)"

    def test_pca_colour_column_splits_traces(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        fig = interactive_pca(coords, color_col=["g1", "g1", "g2", "g2"])

        assert {t.name for t in fig.data} == {"g1", "g2"}

    def test_pca_labels_reach_the_hover_text(self):
        fig = interactive_pca(np.array([[0.0, 0.0], [1.0, 1.0]]), labels=["s1", "s2"])

        assert list(fig.data[0].text) == ["s1", "s2"]

    def test_manhattan_one_trace_per_chromosome(self):
        fig = interactive_manhattan(["1", "1", "2", "2", "2"],
                                    [100, 200, 50, 60, 70],
                                    [3.0, 4.0, 2.0, 6.0, 1.0])

        assert [t.name for t in fig.data] == ["1", "2"]
        assert points(fig.data[1]) == [(50, 2.0), (60, 6.0), (70, 1.0)]

    def test_manhattan_threshold_adds_a_significance_line(self):
        fig = interactive_manhattan(["1", "1"], [10, 20], [3.0, 4.0], threshold=5e-8)

        assert len(list(fig.layout.shapes)) == 1
        assert round(fig.layout.shapes[0].y0, 3) == round(-np.log10(5e-8), 3)

    def test_ma_splits_on_the_fold_change_threshold(self):
        fig = interactive_ma(np.array([1.0, 2.0, 3.0, 4.0]),
                             np.array([0.1, 1.5, -0.2, 2.0]))

        not_sig, sig = fig.data
        assert (not_sig.name, sig.name) == ("Not significant", "Significant")
        assert points(sig) == [(2.0, 1.5), (4.0, 2.0)]
        assert points(not_sig) == [(1.0, 0.1), (3.0, -0.2)]

    def test_qq_pairs_observed_with_expected_quantiles(self):
        pvals = np.clip(rng.random(30), 1e-6, 1 - 1e-6)
        observed, reference = interactive_qq(pvals).data

        n = len(pvals)
        assert np.allclose(np.asarray(observed.x, dtype=float),
                           stats.norm.ppf(np.arange(1, n + 1) / (n + 1)))
        assert np.allclose(np.asarray(observed.y, dtype=float),
                           stats.norm.ppf(np.sort(pvals)))
        assert reference.mode == "lines"

    def test_dotplot_sizes_colours_and_genes(self):
        fig = interactive_dotplot(["c1", "c2", "c1", "c2"],
                                  ["g1", "g2", "g1", "g2"],
                                  [0.1, 0.2, 0.3, 0.4], sizes=[10, 20, 30, 40])
        trace = fig.data[0]

        assert list(trace.x) == ["c1", "c2", "c1", "c2"]
        assert list(trace.y) == ["g1", "g2", "g1", "g2"]
        assert list(trace.marker.size) == [10, 20, 30, 40]
        assert list(trace.marker.color) == [0.1, 0.2, 0.3, 0.4]
        assert "Expr: 0.10" in trace.text[0]

    def test_upset_sunburst_lists_intersections_then_sets(self):
        trace = interactive_upset_sunburst({"A": 10, "B": 5}, {"A&B": 2}).data[0]

        assert list(trace.labels) == ["A&B", "A", "B"]
        assert list(trace.values) == [2, 10, 5]


class TestExportReport:
    def test_every_named_plot_appears_in_the_html(self, tmp_path):
        out = tmp_path / "report.html"
        plots = {"scatter": interactive_scatter([1, 2], [3, 4]),
                 "bar": interactive_bar(["a", "b"], [1, 2])}
        export_interactive_report(plots, str(out))

        html = out.read_text(encoding="utf-8")
        assert "BioSuite Interactive Report" in html
        for name in plots:
            assert f"<h2>{name}</h2>" in html
        assert html.count("plotly-graph-div") == len(plots)


# The empty-input contract (refused by name, by both backends) lives in
# ``test_interactive_plots_refuse_an_empty_table.py``: it replaced a pin that
# held the two backends to different answers for the same call.


# ── the file each function promises when it is handed a path ─────────────────

HTML_CALLS = [
    ("scatter", lambda p: interactive_scatter([1, 2], [3, 4], output_html=p)),
    ("bar", lambda p: interactive_bar(["a", "b"], [1, 2], output_html=p)),
    ("line", lambda p: interactive_line([1, 2], [[3, 4]], output_html=p)),
    ("scatter3d", lambda p: interactive_3d_scatter([1, 2], [3, 4], [5, 6], output_html=p)),
    ("pie", lambda p: interactive_pie(["a", "b"], [1, 2], output_html=p)),
    ("boxplot", lambda p: interactive_boxplot({"a": [1, 2], "b": [3, 4]}, output_html=p)),
    ("violin", lambda p: interactive_violin({"a": [1, 2], "b": [3, 4]}, output_html=p)),
    ("heatmap", lambda p: interactive_heatmap([[1.0, 2.0], [3.0, 4.0]], output_html=p)),
    ("volcano", lambda p: interactive_volcano([1.5, -2.0], [0.01, 0.4], output_html=p)),
    ("pca", lambda p: interactive_pca(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                                      output_html=p)),
    ("manhattan", lambda p: interactive_manhattan(["c1", "c1"], [10, 20], [5.0, 3.0],
                                                  output_html=p)),
    ("ma", lambda p: interactive_ma([1.0, 2.0], [0.5, 2.0],
                                    sig=np.array([False, True]), output_html=p)),
    ("qq", lambda p: interactive_qq([0.1, 0.4, 0.9], output_html=p)),
    ("dotplot", lambda p: interactive_dotplot(["c1", "c2"], ["g1", "g2"], [0.1, 0.2],
                                              output_html=p)),
    ("upset", lambda p: interactive_upset_sunburst({"A": 10, "B": 5}, {"A&B": 2},
                                                   output_html=p)),
]


@pytest.mark.parametrize("name,call", HTML_CALLS, ids=[n for n, _ in HTML_CALLS])
def test_every_plot_writes_the_html_file_it_was_asked_for(name, call, tmp_path):
    """``output_html`` is one argument on every signature, so it is one promise on
    every function — and each of them had to be measured on its own."""
    target = tmp_path / f"{name}.html"

    fig = call(str(target))

    assert isinstance(fig, go.Figure)
    assert target.exists(), f"{name} took output_html and wrote nothing"
    assert target.stat().st_size > 1000, f"{name}'s HTML has no figure in it"
