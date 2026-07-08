"""
Comprehensive tests for the 4 untested plotting modules:
  1. biosuite.plotting.biological_plots
  2. biosuite.plotting.math_plots
  3. biosuite.plotting.network_plots
  4. biosuite.plotting.specialized_plots
"""
import os
import sys
import pytest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs():
    """Close all matplotlib figures after every test."""
    yield
    plt.close('all')


@pytest.fixture()
def _quiet_config():
    """Temporarily set config['quiet'] = True so safe_float_input returns defaults."""
    from biosuite.core.utils import config, session
    old_quiet = config.get('quiet')
    config['quiet'] = True
    old_session = dict(session)
    session.clear()
    yield config
    config['quiet'] = old_quiet
    session.update(old_session)


def _input_factory(default='n'):
    """Return a mock for builtins.input that handles common prompts."""
    def _mock_input(prompt=''):
        p = prompt.lower()
        if 'number of sets' in p:
            return '2'
        if 'use default' in p:
            return 'y'
        return default
    return _mock_input


# ============================================================================
# 1. BIOLOGICAL PLOTS
# ============================================================================

class TestVolcanoPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_volcano_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import volcano_plot
        volcano_plot()
        mock_show.assert_called()

    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_volcano_plot_default_data(self, mock_input, mock_show, _quiet_config):
        """Verify volcano_plot runs with default synthetic data."""
        from biosuite.plotting.biological_plots import volcano_plot
        volcano_plot()
        # Should have created a figure and called show
        mock_show.assert_called_once()


class TestPCAPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_pca_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        try:
            from biosuite.plotting.biological_plots import pca_plot
        except ImportError:
            pytest.skip("sklearn not installed")
        pca_plot()
        mock_show.assert_called()


class TestManhattanPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_manhattan_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import manhattan_plot
        manhattan_plot()
        mock_show.assert_called()


class TestMAPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_ma_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import ma_plot
        ma_plot()
        mock_show.assert_called()


class TestVennDiagram:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_venn_diagram_2sets(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import venn_diagram
        venn_diagram()
        mock_show.assert_called()


class TestBarplotCustom:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_barplot_custom_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import barplot_custom
        barplot_custom()
        mock_show.assert_called()


class TestBoxplotCustom:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_boxplot_custom_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import boxplot_custom
        boxplot_custom()
        mock_show.assert_called()


class TestHeatmapCustom:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_heatmap_custom_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import heatmap_custom
        heatmap_custom()
        mock_show.assert_called()


class TestScatterCustom:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_scatter_custom_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import scatter_custom
        scatter_custom()
        mock_show.assert_called()


class TestTimeseriesPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_timeseries_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import timeseries_plot
        timeseries_plot()
        mock_show.assert_called()


class TestQQPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_qq_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import qq_plot
        qq_plot()
        mock_show.assert_called()


class TestViolinPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_violin_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.biological_plots import violin_plot
        violin_plot()
        mock_show.assert_called()


# ============================================================================
# 1b. BIOLOGICAL PLOTS – internal drawing helpers (non-interactive)
# ============================================================================

class TestDrawVenn2:
    def test_draw_venn2_returns_ax(self):
        from biosuite.plotting.biological_plots import draw_venn2
        ax = draw_venn2([10, 15, 4], set_labels=('A', 'B'))
        assert ax is not None

    def test_draw_venn2_custom_labels(self):
        from biosuite.plotting.biological_plots import draw_venn2
        ax = draw_venn2([5, 8, 3], set_labels=('Set1', 'Set2'))
        assert ax is not None


class TestDrawVenn3:
    def test_draw_venn3_returns_ax(self):
        from biosuite.plotting.biological_plots import draw_venn3
        ax = draw_venn3([8, 8, 8, 3, 3, 3, 1])
        assert ax is not None

    def test_draw_venn3_custom_labels(self):
        from biosuite.plotting.biological_plots import draw_venn3
        ax = draw_venn3([10, 10, 10, 5, 5, 5, 2], set_labels=('X', 'Y', 'Z'))
        assert ax is not None


class TestDrawMotifLogo:
    def test_draw_motif_logo_returns_ax(self):
        from biosuite.plotting.biological_plots import draw_motif_logo
        seqs = ['AAGT', 'AAGT', 'CAGT', 'AACT', 'ACGT', 'ATGT']
        ax = draw_motif_logo(seqs)
        assert ax is not None

    def test_draw_motif_logo_single_seq(self):
        from biosuite.plotting.biological_plots import draw_motif_logo
        ax = draw_motif_logo(['ACGTACGT'])
        assert ax is not None


class TestDrawSankey:
    def test_draw_sankey_returns_ax(self):
        from biosuite.plotting.biological_plots import draw_sankey
        labels = ['A', 'B', 'C', 'D']
        sources = [0, 1, 2]
        targets = [1, 2, 3]
        values = [10, 20, 30]
        ax = draw_sankey(labels, sources, targets, values)
        assert ax is not None


# ============================================================================
# 2. MATH PLOTS
# ============================================================================

class TestSinePlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_sine_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import sine_plot
        sine_plot()
        mock_show.assert_called()


class TestCosinePlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_cosine_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import cosine_plot
        cosine_plot()
        mock_show.assert_called()


class TestLinearPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_linear_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import linear_plot
        linear_plot()
        mock_show.assert_called()


class TestQuadraticPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_quadratic_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import quadratic_plot
        quadratic_plot()
        mock_show.assert_called()


class TestCubicPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_cubic_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import cubic_plot
        cubic_plot()
        mock_show.assert_called()


class TestExponentialPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_exponential_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import exponential_plot
        exponential_plot()
        mock_show.assert_called()


class TestLogisticPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', return_value='n')
    def test_logistic_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.math_plots import logistic_plot
        logistic_plot()
        mock_show.assert_called()


# ============================================================================
# 3. NETWORK PLOTS
# ============================================================================

class TestNetworkPlots:
    """Test network creation + plotting.  The public API is
    create_ppi_network / create_regulatory_network + plot_network."""

    @pytest.fixture(autouse=True)
    def _require_networkx(self):
        try:
            import networkx  # noqa: F401
        except ImportError:
            pytest.skip("networkx not installed")

    # --- PPI ---
    def test_create_ppi_network(self):
        from biosuite.plotting.network_plots import create_ppi_network
        interactions = [
            ('ProtA', 'ProtB', 0.9),
            ('ProtB', 'ProtC', 0.7),
            ('ProtA', 'ProtC', 0.5),
        ]
        G = create_ppi_network(interactions)
        assert G is not None
        assert len(G) == 3
        assert G.number_of_edges() == 3

    def test_create_ppi_network_single_edge(self):
        from biosuite.plotting.network_plots import create_ppi_network
        G = create_ppi_network([('X', 'Y', 1.0)])
        assert G is not None
        assert len(G) == 2

    def test_plot_ppi_network(self):
        """Build a PPI graph and plot it → should return a fig."""
        from biosuite.plotting.network_plots import create_ppi_network, plot_network
        interactions = [
            ('A', 'B', 0.8), ('B', 'C', 0.6),
            ('C', 'D', 0.4), ('A', 'D', 0.9),
        ]
        G = create_ppi_network(interactions)
        fig = plot_network(G, title='Test PPI')
        assert fig is not None
        plt.close(fig)

    # --- Regulatory ---
    def test_create_regulatory_network(self):
        from biosuite.plotting.network_plots import create_regulatory_network
        edges = [
            ('TF1', 'GeneA', 'activation'),
            ('TF1', 'GeneB', 'repression'),
            ('TF2', 'GeneC', 'activation'),
        ]
        G = create_regulatory_network(edges)
        assert G is not None
        assert len(G) == 5
        assert G.is_directed()

    def test_create_regulatory_network_empty(self):
        from biosuite.plotting.network_plots import create_regulatory_network
        G = create_regulatory_network([])
        assert G is not None
        assert len(G) == 0

    def test_plot_regulatory_network(self):
        from biosuite.plotting.network_plots import create_regulatory_network, plot_network
        edges = [
            ('TF1', 'GeneA', 'activation'),
            ('TF2', 'GeneB', 'repression'),
            ('GeneA', 'GeneB', 'activation'),
        ]
        G = create_regulatory_network(edges)
        fig = plot_network(G, title='Test Regulatory')
        assert fig is not None
        plt.close(fig)

    # --- plot_network edge cases ---
    def test_plot_network_empty_graph(self):
        from biosuite.plotting.network_plots import plot_network
        import networkx as nx
        G = nx.Graph()
        fig = plot_network(G, title='Empty')
        assert fig is not None
        plt.close(fig)

    def test_plot_network_none(self):
        from biosuite.plotting.network_plots import plot_network
        fig = plot_network(None, title='None graph')
        assert fig is not None
        plt.close(fig)

    def test_network_statistics(self):
        from biosuite.plotting.network_plots import create_ppi_network, network_statistics
        interactions = [('A', 'B', 1), ('B', 'C', 1), ('C', 'D', 1)]
        G = create_ppi_network(interactions)
        stats = network_statistics(G)
        assert stats['nodes'] == 4
        assert stats['edges'] == 3
        assert 'density' in stats
        assert 'avg_degree' in stats

    def test_network_statistics_empty(self):
        from biosuite.plotting.network_plots import network_statistics
        stats = network_statistics(None)
        assert stats == {}

    def test_plot_degree_distribution(self):
        from biosuite.plotting.network_plots import create_ppi_network, plot_degree_distribution
        interactions = [('A', 'B', 1), ('B', 'C', 1), ('C', 'D', 1), ('A', 'D', 1)]
        G = create_ppi_network(interactions)
        fig = plot_degree_distribution(G)
        assert fig is not None
        plt.close(fig)

    def test_create_metabolic_network(self):
        from biosuite.plotting.network_plots import create_metabolic_network
        reactions = [
            ('Glucose', 'G6P', 'HK'),
            ('G6P', 'F6P', 'PGI'),
        ]
        G = create_metabolic_network(reactions)
        assert G is not None
        assert G.is_directed()
        assert len(G) == 3


# ============================================================================
# 4. SPECIALIZED PLOTS
# ============================================================================

class TestGSEAPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_gsea_plot_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.specialized_plots import gsea_plot
        gsea_plot()
        mock_show.assert_called()


class TestMotifLogo:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('y'))
    def test_motif_logo_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.specialized_plots import motif_logo
        motif_logo()
        mock_show.assert_called()


class TestSankeyDiagram:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('y'))
    def test_sankey_diagram_returns_fig(self, mock_input, mock_show, _quiet_config):
        from biosuite.plotting.specialized_plots import sankey_diagram
        sankey_diagram()
        mock_show.assert_called()


class TestUMAPPlot:
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('builtins.input', side_effect=_input_factory('n'))
    def test_umap_plot(self, mock_input, mock_show, _quiet_config):
        """UMAP plot – skips gracefully if umap-learn is not installed."""
        from biosuite.plotting.specialized_plots import HAS_UMAP
        if not HAS_UMAP:
            pytest.skip("umap-learn not installed")
        from biosuite.plotting.specialized_plots import umap_plot
        umap_plot()
        mock_show.assert_called()

    def test_umap_plot_missing_package(self):
        """When HAS_UMAP is False, umap_plot should return without error."""
        from biosuite.plotting import specialized_plots
        old = specialized_plots.HAS_UMAP
        specialized_plots.HAS_UMAP = False
        try:
            specialized_plots.umap_plot()
        finally:
            specialized_plots.HAS_UMAP = old


# ============================================================================
# EXTRA: verify module-level imports & helpers work
# ============================================================================

class TestModuleImports:
    def test_biological_plots_importable(self):
        import biosuite.plotting.biological_plots as bp
        assert hasattr(bp, 'volcano_plot')
        assert hasattr(bp, 'pca_plot')
        assert hasattr(bp, 'manhattan_plot')
        assert hasattr(bp, 'ma_plot')
        assert hasattr(bp, 'venn_diagram')
        assert hasattr(bp, 'barplot_custom')
        assert hasattr(bp, 'boxplot_custom')
        assert hasattr(bp, 'heatmap_custom')
        assert hasattr(bp, 'scatter_custom')
        assert hasattr(bp, 'timeseries_plot')
        assert hasattr(bp, 'qq_plot')
        assert hasattr(bp, 'violin_plot')

    def test_math_plots_importable(self):
        import biosuite.plotting.math_plots as mp
        for name in ('sine_plot', 'cosine_plot', 'linear_plot',
                      'quadratic_plot', 'cubic_plot', 'exponential_plot',
                      'logistic_plot'):
            assert hasattr(mp, name), f"Missing {name}"

    def test_network_plots_importable(self):
        import biosuite.plotting.network_plots as np_mod
        for name in ('create_ppi_network', 'create_regulatory_network',
                      'create_metabolic_network', 'plot_network',
                      'network_statistics'):
            assert hasattr(np_mod, name), f"Missing {name}"

    def test_specialized_plots_importable(self):
        import biosuite.plotting.specialized_plots as sp
        for name in ('gsea_plot', 'motif_logo', 'sankey_diagram', 'umap_plot'):
            assert hasattr(sp, name), f"Missing {name}"

    def test_plotting_package_importable(self):
        import biosuite.plotting
        assert biosuite.plotting is not None
