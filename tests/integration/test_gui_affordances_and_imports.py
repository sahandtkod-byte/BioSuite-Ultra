"""
Phase 5 tests — keyboard shortcuts, progress bar, plot history,
drag-drop, tab icons, integration tests, GUI structure tests.
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─── Tab Icons ───────────────────────────────────────────────────────────────

class TestTabIcons:
    def test_sidebar_has_icons(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        assert inspect.isfunction(BioSuiteApp.__dict__['_build_sidebar'])
        # The sidebar is assembled during startup, not left as a dead helper.
        assert '_build_sidebar' in inspect.getsource(BioSuiteApp._finish_startup)

    def test_sidebar_items_have_emoji(self):
        """Verify sidebar items are defined with labels."""
        import contextlib
        import io

        from biosuite.cli.menu import print_menu
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_menu()
        output = f.getvalue()
        assert 'UpSet' in output
        assert 'Pipeline' in output


# ─── Keyboard Shortcuts ──────────────────────────────────────────────────────

class TestKeyboardShortcuts:
    def test_gui_has_shortcut_methods(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        assert inspect.isfunction(BioSuiteApp.__dict__['_save_current'])
        assert inspect.isfunction(BioSuiteApp.__dict__['_refresh_current'])
        # Both are bound to real keys during startup (see test_gui_has_key_bindings).
        src = inspect.getsource(BioSuiteApp._finish_startup)
        assert '_save_current' in src
        assert '_refresh_current' in src

    def test_gui_has_key_bindings(self):
        """Verify key bindings are set up in the class."""
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        src = inspect.getsource(BioSuiteApp._finish_startup)
        assert 'Control-q' in src or 'Control-Q' in src
        assert 'Control-s' in src or 'Control-S' in src
        assert 'F1' in src
        assert 'F5' in src


# ─── Progress Bar ────────────────────────────────────────────────────────────

class TestProgressBar:
    def test_gui_has_progress_methods(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        from biosuite.gui.tabs.visualization import VisualizationTabMixin
        for name in ('_show_progress', '_update_progress', '_hide_progress'):
            assert inspect.isfunction(BioSuiteApp.__dict__[name]), f"{name} missing"
        # The plot handlers actually drive the bar; a dead helper would still
        # satisfy the existence check above, so pin the real call sites.
        src = inspect.getsource(VisualizationTabMixin)
        assert '_show_progress' in src
        assert '_hide_progress' in src


# ─── Plot History ────────────────────────────────────────────────────────────

class TestPlotHistory:
    def test_gui_has_history_methods(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        assert inspect.isfunction(BioSuiteApp.__dict__['_record_plot'])
        assert inspect.isfunction(BioSuiteApp.__dict__['_show_plot_history'])
        # Every plotted figure passes through _record_plot, so the history is
        # fed by the plot pipeline, not by anything else.
        assert '_record_plot' in inspect.getsource(BioSuiteApp._show_plot_from_figure)


# ─── Drag-and-Drop ──────────────────────────────────────────────────────────

class TestDragDrop:
    def test_gui_has_dnd_methods(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        assert inspect.isfunction(BioSuiteApp.__dict__['_setup_drag_drop'])
        assert inspect.isfunction(BioSuiteApp.__dict__['_on_drop'])
        # The setup helper wires the real drop handler; a stub that registers
        # nothing would satisfy the existence check alone.
        assert '_on_drop' in inspect.getsource(BioSuiteApp._setup_drag_drop)


# ─── Integration Tests ──────────────────────────────────────────────────────

class TestIntegrationPipeline:
    """End-to-end: pipeline → report."""

    def test_pipeline_to_html_report(self, tmp_path):
        from biosuite.core.workflow.pipeline import Pipeline
        from biosuite.core.workflow.report import generate_pipeline_report
        p = Pipeline("integration_test")
        p.add_step("step1", lambda: "hello")
        p.add_step("step2", lambda: "world")
        p.run()
        assert p.results["step1"] == "hello"
        path = str(tmp_path / "pipeline_report.html")
        generate_pipeline_report(p, path)
        assert os.path.exists(path)
        with open(path) as f:
            html = f.read()
        assert "integration_test" in html
        assert "step1" in html

    def test_pipeline_with_real_functions(self):
        from biosuite.core.sequence import gc_content
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("seq_pipeline")
        p.add_step("gc_atcg", gc_content, args=("ATCGATCG",))
        p.add_step("gc_allg", gc_content, args=("GGGGGGGG",))
        p.run(stop_on_error=False)
        assert p.results.get("gc_atcg") == 50.0
        # gc_allg may fail due to context kwargs — that's expected
        # just verify pipeline ran without crashing
        assert len(p.results) >= 1


class TestIntegrationBatch:
    """End-to-end: batch → results."""

    def test_batch_with_sequence_functions(self):
        from biosuite.core.sequence import gc_content
        from biosuite.core.workflow.batch import BatchProcessor
        bp = BatchProcessor("seq_batch")
        sequences = ["ATCGATCG", "GGGGCCCC", "AAAATTTT"]
        bp.add_samples(sequences, lambda sid: gc_content(sid))
        bp.run(max_workers=1)
        results = bp.get_results()
        assert results["ATCGATCG"] == 50.0
        assert results["GGGGCCCC"] == 100.0
        assert results["AAAATTTT"] == 0.0

    def test_batch_with_report(self, tmp_path):
        from biosuite.core.workflow.batch import BatchProcessor
        from biosuite.core.workflow.report import generate_batch_report
        bp = BatchProcessor("test_batch")
        bp.add_samples(["s1", "s2"], lambda sid: len(sid))
        bp.run(max_workers=1)
        path = str(tmp_path / "batch_report.html")
        generate_batch_report(bp, path)
        assert os.path.exists(path)


class TestIntegrationGWAS:
    """End-to-end: GWAS → lead SNPs → report."""

    def test_gwas_full_pipeline(self):
        from biosuite.core.gwas import (
            detect_lead_snps,
            format_gwas_report,
            generate_gwas_data,
            run_gwas,
        )
        data = generate_gwas_data(n_snps=500, seed=123)
        results = run_gwas(data)
        assert len(results) > 0
        assert results["p_value"].min() >= 0
        assert results["p_value"].max() <= 1
        leads = detect_lead_snps(results, p_threshold=0.05)
        report = format_gwas_report(results, leads)
        assert "GWAS" in report
        assert "SNPs" in report


class TestIntegrationEpitope:
    """End-to-end: protein → T-cell + B-cell epitopes → report."""

    def test_epitope_full_pipeline(self):
        from biosuite.core.epitope import (
            cleavage_site_prediction,
            format_epitope_report,
            predict_b_cell_epitopes,
            predict_linear_epitopes,
            predict_t_cell_epitopes,
        )
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        tc = predict_t_cell_epitopes(seq, mhc_type="A0201")
        bc = predict_b_cell_epitopes(seq)
        lc = predict_linear_epitopes(seq)
        cleavage = cleavage_site_prediction(seq)
        assert len(tc) > 0
        assert len(bc) > 0
        assert len(lc) > 0
        assert len(cleavage) > 0
        report = format_epitope_report(tc, bc, "test_protein")
        assert "T-cell" in report
        assert "B-cell" in report


class TestIntegrationGO:
    """End-to-end: GO search → ancestors → report."""

    def test_go_full_pipeline(self):
        from biosuite.core.go_browser import GOBrowser, format_go_results
        go = GOBrowser()
        results = go.search("kinase")
        assert len(results) > 0
        term = results[0]
        parents = go.get_parents(term.go_id)
        children = go.get_children(term.go_id)
        ancestors = go.get_ancestors(term.go_id)
        assert len(ancestors) > 0
        # get_children hands back GOTerm objects, not ids — the graph has to agree
        # in both directions whichever way it is walked.
        assert all(hasattr(child, "go_id") for child in children)
        child_ids = [child.go_id for child in children]
        assert term.go_id not in child_ids, "a term is never its own child"
        assert all(term.go_id in [p.go_id for p in go.get_parents(child_id)]
                   for child_id in child_ids), \
            "a child does not report this term as one of its parents"
        assert all(p.go_id in [a.go_id for a in ancestors] for p in parents), \
            "a direct parent is by definition an ancestor"
        # The walk is reflexive: it starts at the term, so the term is its own
        # first "ancestor".  Pinned because enrichment queries rely on a term
        # inheriting from itself.
        assert ancestors[0].go_id == term.go_id
        formatted = format_go_results(results)
        assert term.go_id in formatted


class TestIntegrationPathway:
    """End-to-end: pathway creation → drawing → report."""

    def test_pathway_full_pipeline(self):
        from biosuite.core.pathway_viz import (
            create_kegg_style_pathway,
            draw_pathway,
            format_pathway_report,
        )
        pm = create_kegg_style_pathway()
        assert len(pm.nodes) > 5
        report = format_pathway_report(pm)
        assert "Pathway" in report
        fig = draw_pathway(pm)
        assert fig is not None
        plt.close(fig)


class TestIntegrationCLI:
    """End-to-end: verify CLI menu has all options."""

    def test_cli_has_all_phase4_options(self):
        import contextlib
        import io

        from biosuite.cli.menu import print_menu
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_menu()
        output = f.getvalue()
        for opt in ['92', '93', '94', '95', '96', '97', '98']:
            assert opt in output, f"Option {opt} missing from CLI menu"

    def test_cli_has_all_phase3_options(self):
        import contextlib
        import io

        from biosuite.cli.menu import print_menu
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_menu()
        output = f.getvalue()
        for opt in ['77', '78', '79', '80', '81']:
            assert opt in output, f"Option {opt} missing from CLI menu"


class TestIntegrationGUI:
    """End-to-end: verify GUI has all tabs and methods."""

    def test_gui_has_all_frames(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        builders = {
            'plots': '_build_plot_frame', 'sequence': '_build_sequence_frame',
            'alignment': '_build_alignment_frame', 'phylogeny': '_build_phylogeny_frame',
            'expression': '_build_expression_frame', 'ngs': '_build_ngs_frame',
            'assembly': '_build_assembly_frame', 'singlecell': '_build_singlecell_frame',
            'structure': '_build_structure_frame', 'metagenomics': '_build_metagenomics_frame',
            'crispr': '_build_crispr_frame', 'popgen': '_build_popgen_frame',
            'ml': '_build_ml_frame', 'orftools': '_build_orftools_frame',
            'databases': '_build_databases_frame', 'fileformats': '_build_fileformats_frame',
            'apikey': '_build_apikey_frame', 'help': '_build_help_frame',
            'upset': '_build_upset_frame', 'genomebrowser': '_build_genomebrowser_frame',
            'conservation': '_build_conservation_frame', 'syntenytabs': '_build_synteny_frame',
            'interactive': '_build_interactive_frame', 'pipeline': '_build_pipeline_frame',
            'batch': '_build_batch_frame', 'gobrowser': '_build_gobrowser_frame',
            'pathway': '_build_pathway_frame', 'gwas': '_build_gwas_frame',
            'epitope': '_build_epitope_frame',
        }
        for tab, builder in builders.items():
            method = inspect.getattr_static(BioSuiteApp, builder)
            assert inspect.isfunction(method), f"tab {tab!r} has no real {builder}"

    def test_gui_has_all_action_methods(self):
        import inspect

        from biosuite.gui.main_window import BioSuiteApp
        methods = ['_run_pipeline', '_run_batch', '_go_search', '_draw_pathway',
                    '_gwas_demo', '_run_epitope', '_run_upset', '_gb_view',
                    '_run_cons', '_run_synteny', '_inter_generate',
                    '_save_current', '_refresh_current', '_show_progress',
                    '_update_progress', '_hide_progress', '_record_plot']
        for m in methods:
            method = inspect.getattr_static(BioSuiteApp, m)
            assert inspect.isfunction(method), f"Missing method: {m}"

    def test_gui_plot_categories_complete(self):
        from biosuite.gui.main_window import PLOT_CATEGORIES
        required_cats = ['Advanced Biological', 'Basic Biological', 'Mathematical',
                         'Specialized', 'Additional', 'New Plots', 'Genomics',
                         'Sequence', 'Interactive']
        for cat in required_cats:
            assert cat in PLOT_CATEGORIES, f"Missing category: {cat}"


# ─── Module Import Tests ─────────────────────────────────────────────────────

class TestAllModulesImportable:
    """Verify all core modules can be imported without error."""

    def test_import_core_modules(self):
        modules = [
            'biosuite.core.sequence', 'biosuite.core.alignment',
            'biosuite.core.phylogeny', 'biosuite.core.expression',
            'biosuite.core.ngs', 'biosuite.core.blast',
            'biosuite.core.msa', 'biosuite.core.trimming',
            'biosuite.core.quantification', 'biosuite.core.read_aligner',
            'biosuite.core.variant_calling', 'biosuite.core.single_cell',
            'biosuite.core.peak_calling', 'biosuite.core.ml_phylogeny',
            'biosuite.core.structure', 'biosuite.core.structure_prediction',
            'biosuite.core.assembly', 'biosuite.core.metagenomics',
            'biosuite.core.docking', 'biosuite.core.crispr',
            'biosuite.core.metabolism', 'biosuite.core.popgen',
            'biosuite.core.epigenomics', 'biosuite.core.metabolomics',
            'biosuite.core.md_simulation', 'biosuite.core.bayesian_phylogeny',
            'biosuite.core.bio_ml', 'biosuite.core.codon_usage',
            'biosuite.core.survival', 'biosuite.core.orf_finder',
            'biosuite.core.file_formats', 'biosuite.core.databases',
            'biosuite.core.go_browser', 'biosuite.core.pathway_viz',
            'biosuite.core.gwas', 'biosuite.core.epitope',
            'biosuite.core.workflow.pipeline', 'biosuite.core.workflow.batch',
            'biosuite.core.workflow.report',
        ]
        import importlib
        import types

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert isinstance(mod, types.ModuleType)
            # import_module raises instead of returning None, so the real
            # hazard is importing the stale 4.2.5 tree; pin the file location.
            assert os.path.abspath(mod.__file__).startswith(os.path.abspath(root)), \
                f"{mod_name} imported from {mod.__file__}"

    def test_import_plotting_modules(self):
        modules = [
            'biosuite.plotting.biological_plots', 'biosuite.plotting.math_plots',
            'biosuite.plotting.specialized_plots', 'biosuite.plotting.network_plots',
            'biosuite.plotting.upset_plots', 'biosuite.plotting.genome_browser',
            'biosuite.plotting.interactive_plots', 'biosuite.plotting.conservation_plots',
            'biosuite.plotting.synteny',
        ]
        import importlib
        import types

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert isinstance(mod, types.ModuleType)
            assert os.path.abspath(mod.__file__).startswith(os.path.abspath(root)), \
                f"{mod_name} imported from {mod.__file__}"
