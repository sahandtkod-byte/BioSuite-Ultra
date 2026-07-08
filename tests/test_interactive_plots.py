"""
Comprehensive tests for biosuite.plotting.interactive_plots module.

Tests all interactive plot functions with mock data, verifies HTML export,
and checks graceful handling of edge cases (empty/invalid input).
"""
import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock

# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for output files; clean up after test."""
    d = tempfile.mkdtemp(prefix="biosuite_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_fig_to_html(tmp_dir, filename="test_report.html"):
    """Helper to build an output path inside tmp_dir."""
    return os.path.join(tmp_dir, filename)


# ── Test interactive_scatter ─────────────────────────────────────────────────

class TestInteractiveScatter:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0]
        fig = interactive_scatter(x, y, title="Test Scatter")
        assert fig is not None

    def test_with_labels(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        x = [1, 2, 3]
        y = [4, 5, 6]
        labels = ["A", "B", "C"]
        fig = interactive_scatter(x, y, labels=labels)
        assert fig is not None

    def test_with_color_col(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        color_col = ["cat1", "cat1", "cat2", "cat2"]
        fig = interactive_scatter(x, y, color_col=color_col)
        assert fig is not None

    def test_with_color_and_labels(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        labels = ["a", "b", "c", "d"]
        color_col = ["X", "X", "Y", "Y"]
        fig = interactive_scatter(x, y, labels=labels, color_col=color_col)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_scatter
        out = _make_fig_to_html(tmp_dir, "scatter.html")
        x = [1, 2, 3]
        y = [4, 5, 6]
        fig = interactive_scatter(x, y, output_html=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_empty_input(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        fig = interactive_scatter([], [])
        assert fig is not None

    def test_single_point(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        fig = interactive_scatter([5.0], [10.0])
        assert fig is not None

    def test_with_numpy_arrays(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig = interactive_scatter(x, y)
        assert fig is not None

    def test_fallback_without_plotly(self):
        """Test matplotlib fallback path when plotly is not available."""
        from biosuite.plotting.interactive_plots import interactive_scatter
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_scatter([1, 2, 3], [4, 5, 6])
            assert fig is not None


# ── Test interactive_bar ─────────────────────────────────────────────────────

class TestInteractiveBar:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        cats = ["A", "B", "C", "D"]
        vals = [10, 25, 15, 30]
        fig = interactive_bar(cats, vals)
        assert fig is not None

    def test_with_errors(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        cats = ["X", "Y"]
        vals = [100, 200]
        errs = [10, 20]
        fig = interactive_bar(cats, vals, errors=errs)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_bar
        out = _make_fig_to_html(tmp_dir, "bar.html")
        fig = interactive_bar(["a", "b"], [1, 2], output_html=out)
        assert os.path.exists(out)

    def test_empty_input(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        fig = interactive_bar([], [])
        assert fig is not None

    def test_single_category(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        fig = interactive_bar(["only"], [42])
        assert fig is not None

    def test_many_categories(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        cats = [f"cat_{i}" for i in range(50)]
        vals = list(range(50))
        fig = interactive_bar(cats, vals)
        assert fig is not None


# ── Test interactive_heatmap ─────────────────────────────────────────────────

class TestInteractiveHeatmap:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        data = np.random.rand(5, 5).tolist()
        fig = interactive_heatmap(data)
        assert fig is not None

    def test_with_labels(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        data = [[1, 2], [3, 4]]
        row_labels = ["Row1", "Row2"]
        col_labels = ["Col1", "Col2"]
        fig = interactive_heatmap(data, row_labels=row_labels, col_labels=col_labels)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        out = _make_fig_to_html(tmp_dir, "heatmap.html")
        data = [[1, 2], [3, 4]]
        fig = interactive_heatmap(data, output_html=out)
        assert os.path.exists(out)

    def test_empty_data(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        fig = interactive_heatmap([[]])
        assert fig is not None

    def test_1x1_heatmap(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        fig = interactive_heatmap([[42.0]])
        assert fig is not None

    def test_with_custom_colorscale(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        data = np.arange(16).reshape(4, 4).tolist()
        fig = interactive_heatmap(data, colorscale='RdBu')
        assert fig is not None


# ── Test interactive_volcano ─────────────────────────────────────────────────

class TestInteractiveVolcano:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        np.random.seed(42)
        lfc = np.random.randn(100)
        pvals = np.random.uniform(0.001, 1, 100)
        fig = interactive_volcano(lfc, pvals)
        assert fig is not None

    def test_with_gene_names(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        lfc = [2.0, -3.0, 0.5, 4.0]
        pvals = [0.001, 0.005, 0.5, 0.0001]
        gene_names = ["BRCA1", "TP53", "MYC", "EGFR"]
        fig = interactive_volcano(lfc, pvals, gene_names=gene_names)
        assert fig is not None

    def test_with_thresholds(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        lfc = np.array([3.0, -2.0, 0.1])
        pvals = np.array([0.01, 0.01, 0.5])
        fig = interactive_volcano(lfc, pvals, fc_thresh=2.0, p_thresh=0.05)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_volcano
        out = _make_fig_to_html(tmp_dir, "volcano.html")
        lfc = [1, -1, 2, -2]
        pvals = [0.01, 0.05, 0.001, 0.1]
        fig = interactive_volcano(lfc, pvals, output_html=out)
        assert os.path.exists(out)

    def test_all_significant(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        lfc = [5.0, -5.0, 4.0]
        pvals = [0.0001, 0.0001, 0.0001]
        fig = interactive_volcano(lfc, pvals, fc_thresh=1.0, p_thresh=0.05)
        assert fig is not None

    def test_none_significant(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        lfc = [0.1, -0.1, 0.05]
        pvals = [0.8, 0.9, 0.7]
        fig = interactive_volcano(lfc, pvals)
        assert fig is not None

    def test_fallback_without_plotly(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            # Use numpy arrays because _fallback_volcano uses numpy boolean indexing
            lfc = np.array([1.0, -1.0])
            pvals = np.array([0.01, 0.1])
            fig = interactive_volcano(lfc, pvals)
            assert fig is not None


# ── Test interactive_line ────────────────────────────────────────────────────

class TestInteractiveLine:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_line
        x = [1, 2, 3, 4, 5]
        ys = [[1, 4, 9, 16, 25], [5, 4, 3, 2, 1]]
        fig = interactive_line(x, ys)
        assert fig is not None

    def test_with_names(self):
        from biosuite.plotting.interactive_plots import interactive_line
        x = [0, 1, 2, 3]
        ys = [[0, 1, 4, 9], [0, -1, -4, -9]]
        names = ["y=x²", "y=-x²"]
        fig = interactive_line(x, ys, names=names)
        assert fig is not None

    def test_single_line(self):
        from biosuite.plotting.interactive_plots import interactive_line
        fig = interactive_line([1, 2, 3], [[10, 20, 30]])
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_line
        out = _make_fig_to_html(tmp_dir, "line.html")
        fig = interactive_line([1, 2, 3], [[4, 5, 6]], output_html=out)
        assert os.path.exists(out)

    def test_empty_data(self):
        from biosuite.plotting.interactive_plots import interactive_line
        fig = interactive_line([], [[]])
        assert fig is not None

    def test_many_series(self):
        from biosuite.plotting.interactive_plots import interactive_line
        x = list(range(20))
        ys = [list(np.random.rand(20)) for _ in range(10)]
        fig = interactive_line(x, ys)
        assert fig is not None


# ── Test interactive_3d_scatter ──────────────────────────────────────────────

class TestInteractive3DScatter:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [1, 2, 3, 4, 5]
        fig = interactive_3d_scatter(x, y, z)
        assert fig is not None

    def test_with_labels(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        x, y, z = [1, 2], [3, 4], [5, 6]
        labels = ["point A", "point B"]
        fig = interactive_3d_scatter(x, y, z, labels=labels)
        assert fig is not None

    def test_with_color_col(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        x = [1, 2, 3, 4]
        y = [1, 2, 3, 4]
        z = [1, 2, 3, 4]
        color_col = ["red", "blue", "red", "blue"]
        fig = interactive_3d_scatter(x, y, z, color_col=color_col)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        out = _make_fig_to_html(tmp_dir, "3d.html")
        fig = interactive_3d_scatter([1, 2], [3, 4], [5, 6], output_html=out)
        assert os.path.exists(out)

    def test_empty_input(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        fig = interactive_3d_scatter([], [], [])
        assert fig is not None

    def test_numpy_arrays(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        pts = np.random.rand(30, 3)
        fig = interactive_3d_scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        assert fig is not None


# ── Test interactive_pie ─────────────────────────────────────────────────────

class TestInteractivePie:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        labels = ["A", "B", "C", "D"]
        values = [30, 25, 20, 25]
        fig = interactive_pie(labels, values)
        assert fig is not None

    def test_two_slices(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        fig = interactive_pie(["Yes", "No"], [60, 40])
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_pie
        out = _make_fig_to_html(tmp_dir, "pie.html")
        fig = interactive_pie(["a", "b"], [1, 2], output_html=out)
        assert os.path.exists(out)

    def test_empty_input(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        fig = interactive_pie([], [])
        assert fig is not None

    def test_single_slice(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        fig = interactive_pie(["only"], [100])
        assert fig is not None

    def test_many_slices(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        n = 20
        fig = interactive_pie([f"slice_{i}" for i in range(n)], list(range(1, n + 1)))
        assert fig is not None


# ── Test export_interactive_report ───────────────────────────────────────────

class TestExportInteractiveReport:
    def test_creates_html_file(self, tmp_dir):
        from biosuite.plotting.interactive_plots import (
            interactive_scatter, interactive_bar, export_interactive_report
        )
        scatter = interactive_scatter([1, 2, 3], [4, 5, 6])
        bar = interactive_bar(["a", "b", "c"], [10, 20, 30])
        report_path = _make_fig_to_html(tmp_dir, "report.html")
        export_interactive_report(
            {"Scatter Plot": scatter, "Bar Chart": bar},
            output_path=report_path
        )
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 500  # meaningful HTML
        with open(report_path, encoding='utf-8') as f:
            content = f.read()
        assert "BioSuite Interactive Report" in content
        assert "Scatter Plot" in content
        assert "Bar Chart" in content

    def test_single_plot_report(self, tmp_dir):
        from biosuite.plotting.interactive_plots import (
            interactive_heatmap, export_interactive_report
        )
        hm = interactive_heatmap([[1, 2], [3, 4]])
        report_path = _make_fig_to_html(tmp_dir, "single.html")
        export_interactive_report({"Heatmap": hm}, output_path=report_path)
        assert os.path.exists(report_path)

    def test_no_plotly_returns_none(self, tmp_dir):
        """When plotly is not installed, export_interactive_report should print warning and return None."""
        from biosuite.plotting.interactive_plots import export_interactive_report
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            result = export_interactive_report({}, output_path="dummy.html")
            assert result is None
            assert not os.path.exists("dummy.html")

    def test_empty_dict(self, tmp_dir):
        from biosuite.plotting.interactive_plots import export_interactive_report
        report_path = _make_fig_to_html(tmp_dir, "empty.html")
        export_interactive_report({}, output_path=report_path)
        assert os.path.exists(report_path)


# ── Test interactive_boxplot ─────────────────────────────────────────────────

class TestInteractiveBoxplot:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        data = {
            "Group A": [10, 12, 11, 13, 10, 14],
            "Group B": [20, 22, 19, 23, 21],
            "Group C": [15, 16, 14, 17],
        }
        fig = interactive_boxplot(data)
        assert fig is not None

    def test_single_group(self):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        fig = interactive_boxplot({"Only": [1, 2, 3, 4, 5]})
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        out = _make_fig_to_html(tmp_dir, "box.html")
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        fig = interactive_boxplot(data, output_html=out)
        assert os.path.exists(out)

    def test_empty_groups(self):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        fig = interactive_boxplot({})
        assert fig is not None

    def test_many_groups(self):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        data = {f"Group_{i}": list(np.random.randn(20)) for i in range(10)}
        fig = interactive_boxplot(data)
        assert fig is not None


# ── Test interactive_pca ─────────────────────────────────────────────────────

class TestInteractivePCA:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        coords = np.random.rand(20, 2)
        fig = interactive_pca(coords)
        assert fig is not None

    def test_with_variance_explained(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        coords = np.random.rand(15, 2)
        ve = [0.45, 0.25, 0.10]
        fig = interactive_pca(coords, variance_explained=ve)
        assert fig is not None

    def test_with_labels(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        coords = np.random.rand(10, 2)
        labels = [f"Sample_{i}" for i in range(10)]
        fig = interactive_pca(coords, labels=labels)
        assert fig is not None

    def test_with_color_col(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        coords = np.random.rand(20, 2)
        color_col = ["tumor"] * 10 + ["normal"] * 10
        fig = interactive_pca(coords, color_col=color_col)
        assert fig is not None

    def test_with_all_options(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        coords = np.random.rand(20, 2)
        ve = [0.6, 0.3]
        labels = [f"S{i}" for i in range(20)]
        color = ["A"] * 10 + ["B"] * 10
        fig = interactive_pca(coords, variance_explained=ve, labels=labels, color_col=color)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_pca
        out = _make_fig_to_html(tmp_dir, "pca.html")
        coords = np.random.rand(10, 2)
        fig = interactive_pca(coords, output_html=out)
        assert os.path.exists(out)

    def test_empty_coords(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        # 0-row array of shape (0, 2)
        coords = np.empty((0, 2))
        fig = interactive_pca(coords)
        assert fig is not None

    def test_single_point(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        fig = interactive_pca(np.array([[1.0, 2.0]]))
        assert fig is not None


# ── Test interactive_manhattan ───────────────────────────────────────────────

class TestInteractiveManhattan:
    def test_basic(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        np.random.seed(42)
        chroms = ["chr1"] * 50 + ["chr2"] * 50
        positions = np.random.randint(1, 1_000_000, 100).tolist()
        pvals = np.random.uniform(0, 1, 100)
        neg_log_p = (-np.log10(pvals)).tolist()
        fig = interactive_manhattan(chroms, positions, neg_log_p)
        assert fig is not None

    def test_with_threshold(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        chroms = ["chr1", "chr1", "chr2"]
        positions = [1000, 2000, 3000]
        neg_log_p = [5.0, 1.0, 8.0]
        fig = interactive_manhattan(chroms, positions, neg_log_p, threshold=0.05)
        assert fig is not None

    def test_output_html(self, tmp_dir):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        out = _make_fig_to_html(tmp_dir, "manhattan.html")
        chroms = ["chr1", "chr1"]
        fig = interactive_manhattan(chroms, [100, 200], [3.0, 4.0], output_html=out)
        assert os.path.exists(out)

    def test_single_chromosome(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        chroms = ["chr1"] * 10
        positions = list(range(10))
        neg_log_p = list(range(10))
        fig = interactive_manhattan(chroms, positions, neg_log_p)
        assert fig is not None

    def test_empty_input(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        fig = interactive_manhattan([], [], [])
        assert fig is not None

    def test_many_chromosomes(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        chroms = [f"chr{i}" for i in range(1, 23) for _ in range(10)]
        positions = list(range(220))
        neg_log_p = list(np.random.uniform(0, 6, 220))
        fig = interactive_manhattan(chroms, positions, neg_log_p)
        assert fig is not None


# ── Test fallback without plotly ─────────────────────────────────────────────

class TestFallbackPaths:
    """Ensure the matplotlib fallback path works for all functions when HAS_PLOTLY=False."""

    def test_scatter_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_scatter
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_scatter([1, 2], [3, 4])
            assert fig is not None

    def test_bar_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_bar
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_bar(["a", "b"], [1, 2])
            assert fig is not None

    def test_heatmap_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_heatmap
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_heatmap([[1, 2], [3, 4]])
            assert fig is not None

    def test_volcano_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_volcano
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            # Use numpy arrays because _fallback_volcano uses numpy boolean indexing
            fig = interactive_volcano(np.array([1.0, -1.0]), np.array([0.01, 0.1]))
            assert fig is not None

    def test_line_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_line
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_line([1, 2, 3], [[4, 5, 6]])
            assert fig is not None

    def test_3d_scatter_fallback_returns_none(self):
        from biosuite.plotting.interactive_plots import interactive_3d_scatter
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_3d_scatter([1], [2], [3])
            assert fig is None  # 3D returns None without plotly

    def test_pie_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_pie
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_pie(["a", "b"], [1, 2])
            assert fig is not None

    def test_boxplot_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_boxplot
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_boxplot({"A": [1, 2, 3]})
            assert fig is not None

    def test_pca_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_pca
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_pca(np.random.rand(10, 2))
            assert fig is not None

    def test_manhattan_fallback(self):
        from biosuite.plotting.interactive_plots import interactive_manhattan
        with patch('biosuite.plotting.interactive_plots.HAS_PLOTLY', False):
            fig = interactive_manhattan(["1", "1"], [100, 200], [3.0, 4.0])
            assert fig is not None


# ── Run with: pytest tests/test_interactive_plots.py -v ──────────────────────
