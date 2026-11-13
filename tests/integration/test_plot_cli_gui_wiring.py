"""
Unit tests for Phase 3 CLI and GUI integrations.
Tests that CLI dispatch and GUI frames load without errors.
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCLIUpSet:
    def test_upset_compute_and_plot(self):
        from biosuite.plotting.upset_plots import plot_upset
        sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}, 'C': {3, 4, 5}}
        fig = plot_upset(sets)
        assert len(fig.axes) == 2
        assert [p.get_height() for p in fig.axes[0].patches] == [1, 1, 1, 1, 1]
        assert [t.get_text() for t in fig.axes[0].texts] == ['1'] * 5
        plt.close(fig)

    def test_upset_statistics(self):
        from biosuite.plotting.upset_plots import compute_set_statistics
        sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}}
        stats = compute_set_statistics(sets)
        assert 'sizes' in stats
        assert 'total_union' in stats
        assert 'pairwise_jaccard' in stats


class TestCLIGenomeBrowser:
    def test_bed_track_creation(self, tmp_path):
        from biosuite.plotting.genome_browser import (
            create_bed_track,
            plot_genome_tracks,
        )
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\tgeneA\t10\nchr1\t300\t500\tgeneB\t20\n")
        track = create_bed_track(str(bed))
        assert track['type'] == 'bed'
        fig = plot_genome_tracks([track])
        assert fig is not None
        plt.close(fig)

    def test_vcf_track_creation(self, tmp_path):
        from biosuite.plotting.genome_browser import (
            create_variant_track,
            plot_genome_tracks,
        )
        vcf = tmp_path / "test.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\t.\tA\tG\t30\tPASS\t.\n"
        )
        track = create_variant_track(str(vcf))
        assert track['type'] == 'variant'
        fig = plot_genome_tracks([track])
        assert fig is not None
        plt.close(fig)


class TestCLIInteractive:
    def _close_fig(self, fig):
        try:
            import plotly.graph_objects as go
            if isinstance(fig, go.Figure):
                return
        except ImportError:
            pass
        plt.close(fig)

    def test_interactive_scatter_demo(self):
        import numpy as np

        from biosuite.plotting.interactive_plots import interactive_scatter
        fig = interactive_scatter(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 3
        assert len(fig.data[0].y) == 3
        self._close_fig(fig)

    def test_interactive_volcano_demo(self):
        import numpy as np

        from biosuite.plotting.interactive_plots import interactive_volcano
        fig = interactive_volcano(np.array([0.5, 2.0, -2.5]), np.array([0.5, 0.01, 0.001]))
        assert len(fig.data) == 2  # significant and not, both drawn
        assert sum(len(t.x) for t in fig.data) == 3
        self._close_fig(fig)


class TestCLIConservation:
    def test_logo_and_conservation(self):
        from biosuite.plotting.conservation_plots import plot_logo_with_conservation
        seqs = ['ACGTACGT'] * 5
        fig = plot_logo_with_conservation(seqs)
        assert len(fig.axes) == 2
        # One patch per letter of the 8-base logo, on both panels.
        assert [len(ax.patches) for ax in fig.axes] == [8, 8]
        plt.close(fig)

    def test_motif_enrichment(self):
        from biosuite.plotting.conservation_plots import plot_motif_enrichment
        fig = plot_motif_enrichment(['ATGCGATG', 'GCGATGCG'], ['ATG', 'GCG'])
        assert len(fig.axes) == 1
        assert len(fig.axes[0].patches) == 2  # one bar per motif
        plt.close(fig)


class TestCLISynteny:
    def test_synteny_dotplot(self):
        from biosuite.plotting.synteny import (
            compute_synteny_score,
            plot_synteny_dotplot,
        )
        g1 = ['A', 'B', 'C', 'D']
        g2 = ['A', 'B', 'C', 'D']
        score, pairs = compute_synteny_score(g1, g2)
        assert score == 1.0
        fig = plot_synteny_dotplot(g1, g2)
        assert fig is not None
        plt.close(fig)

    def test_synteny_reversed(self):
        from biosuite.plotting.synteny import compute_synteny_score
        score, _ = compute_synteny_score(['A', 'B', 'C'], ['C', 'B', 'A'])
        assert score < 0.5


class TestCLIMenuNumbers:
    """Verify CLI menu option numbers don't conflict."""

    def test_menu_compiles(self):
        """The menu module loads, its entry points are real, and it prints a menu."""
        import contextlib
        import importlib
        import inspect
        import io

        mod = importlib.import_module('biosuite.cli.menu')
        assert inspect.isfunction(mod.print_menu)
        assert inspect.isfunction(mod.main_cli)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            mod.print_menu()
        assert len(f.getvalue()) > 5000

    def test_all_new_options_in_menu(self):
        """Check that options 77-81 appear in the menu text."""
        import contextlib
        import io

        from biosuite.cli.menu import print_menu
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_menu()
        output = f.getvalue()
        assert '77' in output
        assert '78' in output
        assert '79' in output
        assert '80' in output
        assert '81' in output
        assert 'UpSet' in output
        assert 'Genome Browser' in output
        assert 'Sequence Logo' in output
        assert 'Synteny' in output


class TestGUIModule:
    """Verify GUI module loads and has the new frame builders."""

    def test_plot_categories_have_new_entries(self):
        from biosuite.gui.main_window import PLOT_CATEGORIES
        assert 'Genomics' in PLOT_CATEGORIES
        assert 'Sequence' in PLOT_CATEGORIES
        assert 'Interactive' in PLOT_CATEGORIES
        assert ('UpSet Plot', 'upset') in PLOT_CATEGORIES['Genomics']
        assert ('Genome Browser', 'genome_browser') in PLOT_CATEGORIES['Genomics']
        assert ('Synteny Dotplot', 'synteny') in PLOT_CATEGORIES['Genomics']
        assert ('Sequence Logo', 'seq_logo') in PLOT_CATEGORIES['Sequence']
        assert ('Interactive Scatter', 'interactive_scatter') in PLOT_CATEGORIES['Interactive']

    def test_gui_has_frame_builders(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        for name in ('_build_upset_frame', '_build_genomebrowser_frame',
                     '_build_conservation_frame', '_build_synteny_frame',
                     '_build_interactive_frame'):
            method = inspect.getattr_static(BioSuiteApp, name)
            assert inspect.isfunction(method), f"{name} missing"

    def test_gui_has_gui_plot_funcs(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        names = ('_gui_upset', '_gui_genome_browser', '_gui_seq_logo',
                 '_gui_conservation_bar', '_gui_interactive_scatter',
                 '_gui_interactive_bar', '_gui_interactive_heatmap',
                 '_gui_interactive_volcano', '_gui_interactive_line',
                 '_gui_interactive_pie', '_gui_synteny')
        for name in names:
            method = inspect.getattr_static(BioSuiteApp, name)
            assert inspect.isfunction(method), f"{name} missing"
        # The plot catalogue actually dispatches to these functions; a dead
        # helper would still satisfy the existence check above.
        src = inspect.getsource(BioSuiteApp._build_plot_funcs)
        for name in names:
            assert name in src, f"{name} not wired into _build_plot_funcs"
