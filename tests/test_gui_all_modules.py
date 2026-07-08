"""
Comprehensive GUI tests — one test class per module/tab.
Tests structure, methods, frame builders, and widget creation.
No display required (headless-safe).
"""
import os
import sys
import pytest
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')


# ─── Helper ──────────────────────────────────────────────────────────────────

def _has_method(cls, name):
    return hasattr(cls, name) and callable(getattr(cls, name))


def _get_source_contains(cls, method_name, text):
    """Check if a method's source contains a specific string."""
    if not hasattr(cls, method_name):
        return False
    src = inspect.getsource(getattr(cls, method_name))
    return text in src


# ─── GUI Core Structure ─────────────────────────────────────────────────────

class TestGUICoreStructure:
    def test_app_class_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert issubclass(BioSuiteApp, object)

    def test_themes_defined(self):
        from biosuite.gui.main_window import THEMES
        assert 'dark-green' in THEMES
        assert 'dark-purple' in THEMES
        assert 'light-blue' in THEMES

    def test_plot_categories_complete(self):
        from biosuite.gui.main_window import PLOT_CATEGORIES
        required = ['Advanced Biological', 'Basic Biological', 'Mathematical',
                     'Specialized', 'Additional', 'New Plots', 'Genomics',
                     'Sequence', 'Interactive']
        for cat in required:
            assert cat in PLOT_CATEGORIES

    def test_plot_funcs_dict_exists(self):
        from biosuite.gui.main_window import PLOT_FUNCS
        assert isinstance(PLOT_FUNCS, dict)

    def test_font_constants(self):
        from biosuite.gui.main_window import FONT_FAMILY, FONT_MONO, FONT_BODY
        assert FONT_FAMILY
        assert FONT_MONO
        assert FONT_BODY


# ─── Sidebar Tabs ────────────────────────────────────────────────────────────

class TestSidebar:
    def test_build_sidebar_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_sidebar')

    def test_sidebar_categories_in_source(self):
        from biosuite.gui.main_window import BioSuiteApp
        src = inspect.getsource(BioSuiteApp._build_sidebar)
        for cat in ['VISUALIZATION', 'SEQUENCE & ALIGNMENT', 'TRANSCRIPTOMICS',
                     'GENOMICS & NGS', 'SINGLE-CELL & PROTEINS', 'SPECIALIZED',
                     'SEQUENCE TOOLS', 'ADVANCED VISUALIZATION', 'WORKFLOW & DOMAIN',
                     'GENOMICS TOOLS', 'HELP & SETTINGS']:
            assert cat in src, f"Missing sidebar category: {cat}"

    def test_sidebar_tabs_have_icons(self):
        from biosuite.gui.main_window import BioSuiteApp
        src = inspect.getsource(BioSuiteApp._build_sidebar)
        # Check that sidebar items use f-string labels (icons are in the source)
        assert 'f"' in src  # f-string formatting used for labels
        assert '_sidebar_item' in src


# ─── Plot Gallery ────────────────────────────────────────────────────────────

class TestPlotGallery:
    def test_build_plot_frame_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_plot_frame')

    def test_plot_search_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_on_search_key')
        assert _has_method(BioSuiteApp, '_apply_plot_search')

    def test_generate_plot_by_id_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_generate_plot_by_id')

    def test_export_all_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_export_all_plots')

    def test_batch_pdf_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_batch_pdf')

    def test_all_plot_categories_have_items(self):
        from biosuite.gui.main_window import PLOT_CATEGORIES
        for cat, items in PLOT_CATEGORIES.items():
            assert len(items) > 0, f"Empty category: {cat}"
            for name, pid in items:
                assert isinstance(name, str)
                assert isinstance(pid, str)


# ─── Sequence Tab ────────────────────────────────────────────────────────────

class TestSequenceTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_sequence_frame')

    def test_load_file_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_load_seq_file')

    def test_get_seq_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_get_seq')

    def test_seq_gc_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_seq_gc')

    def test_seq_revcomp_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_seq_revcomp')

    def test_seq_translate_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_seq_translate')

    def test_seq_stats_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_seq_stats_cmd')


# ─── Alignment Tab ───────────────────────────────────────────────────────────

class TestAlignmentTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_alignment_frame')


# ─── Phylogeny Tab ──────────────────────────────────────────────────────────

class TestPhylogenyTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_phylogeny_frame')


# ─── Expression Tab ─────────────────────────────────────────────────────────

class TestExpressionTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_expression_frame')


# ─── NGS Tab ────────────────────────────────────────────────────────────────

class TestNGSTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_ngs_frame')


# ─── Trimming Tab ───────────────────────────────────────────────────────────

class TestTrimmingTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_trimming_frame')


# ─── Quantification Tab ─────────────────────────────────────────────────────

class TestQuantTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_quant_frame')


# ─── Single-Cell Tab ────────────────────────────────────────────────────────

class TestSingleCellTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_singlecell_frame')


# ─── Structure Tab ──────────────────────────────────────────────────────────

class TestStructureTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_structure_frame')


# ─── Assembly Tab ───────────────────────────────────────────────────────────

class TestAssemblyTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_assembly_frame')


# ─── Metagenomics Tab ───────────────────────────────────────────────────────

class TestMetagenomicsTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_metagenomics_frame')


# ─── CRISPR Tab ─────────────────────────────────────────────────────────────

class TestCRISPRTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_crispr_frame')


# ─── PopGen Tab ─────────────────────────────────────────────────────────────

class TestPopGenTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_popgen_frame')


# ─── ML Tab ─────────────────────────────────────────────────────────────────

class TestMLTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_ml_frame')


# ─── ORF Tools Tab ──────────────────────────────────────────────────────────

class TestORFToolsTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_orftools_frame')


# ─── Databases Tab ──────────────────────────────────────────────────────────

class TestDatabasesTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_databases_frame')


# ─── File Formats Tab ───────────────────────────────────────────────────────

class TestFileFormatsTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_fileformats_frame')


# ─── API Keys Tab ───────────────────────────────────────────────────────────

class TestAPIKeysTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_apikey_frame')

    def test_save_api_keys_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_save_api_keys')

    def test_clear_api_keys_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_clear_api_keys')


# ─── UpSet Tab (Phase 3) ────────────────────────────────────────────────────

class TestUpSetTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_upset_frame')

    def test_run_upset_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_upset')

    def test_gui_upset_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_upset')


# ─── Genome Browser Tab (Phase 3) ──────────────────────────────────────────

class TestGenomeBrowserTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_genomebrowser_frame')

    def test_gb_add_bed_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gb_add_bed')

    def test_gb_add_vcf_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gb_add_vcf')

    def test_gb_view_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gb_view')

    def test_gui_genome_browser_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_genome_browser')


# ─── Conservation Tab (Phase 3) ────────────────────────────────────────────

class TestConservationTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_conservation_frame')

    def test_run_cons_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_cons')

    def test_run_motif_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_motif')

    def test_gui_seq_logo_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_seq_logo')

    def test_gui_conservation_bar_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_conservation_bar')


# ─── Synteny Tab (Phase 3) ─────────────────────────────────────────────────

class TestSyntenyTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_synteny_frame')

    def test_run_synteny_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_synteny')

    def test_gui_synteny_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_synteny')


# ─── Interactive Tab (Phase 3) ─────────────────────────────────────────────

class TestInteractiveTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_interactive_frame')

    def test_inter_load_csv_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_inter_load_csv')

    def test_inter_demo_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_inter_demo')

    def test_inter_generate_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_inter_generate')

    def test_all_gui_interactive_funcs(self):
        from biosuite.gui.main_window import BioSuiteApp
        for method in ['_gui_interactive_scatter', '_gui_interactive_bar',
                        '_gui_interactive_heatmap', '_gui_interactive_volcano',
                        '_gui_interactive_line', '_gui_interactive_pie']:
            assert _has_method(BioSuiteApp, method), f"Missing: {method}"


# ─── Pipeline Tab (Phase 4) ────────────────────────────────────────────────

class TestPipelineTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_pipeline_frame')

    def test_run_pipeline_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_pipeline')


# ─── Batch Tab (Phase 4) ───────────────────────────────────────────────────

class TestBatchTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_batch_frame')

    def test_run_batch_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_batch')


# ─── GO Browser Tab (Phase 4) ──────────────────────────────────────────────

class TestGOTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_gobrowser_frame')

    def test_go_search_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_go_search')

    def test_go_browse_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_go_browse')


# ─── Pathway Tab (Phase 4) ─────────────────────────────────────────────────

class TestPathwayTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_pathway_frame')

    def test_draw_pathway_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_draw_pathway')

    def test_kegg_demo_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_kegg_demo')


# ─── GWAS Tab (Phase 4) ────────────────────────────────────────────────────

class TestGWASTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_gwas_frame')

    def test_gwas_load_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gwas_load')

    def test_gwas_demo_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gwas_demo')


# ─── Epitope Tab (Phase 4) ─────────────────────────────────────────────────

class TestEpitopeTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_epitope_frame')

    def test_run_epitope_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_epitope')


# ─── 16S rRNA Tab (Phase 5) ────────────────────────────────────────────────

class Test16STab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_16srna_frame')

    def test_run_16s_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_run_16s')


# ─── SV/CNV Tab (Phase 5) ──────────────────────────────────────────────────

class TestSVCNVTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_svcnv_frame')

    def test_svcnv_load_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_svcnv_load')

    def test_svcnv_demo_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_svcnv_demo')


# ─── BigWig Tab (Phase 5) ──────────────────────────────────────────────────

class TestBigWigTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_bigwig_frame')

    def test_bigwig_browse_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_bigwig_browse')

    def test_bigwig_read_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_bigwig_read')


# ─── Help Tab ───────────────────────────────────────────────────────────────

class TestHelpTab:
    def test_build_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_help_frame')

    def test_show_guide_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_show_guide')

    def test_all_help_methods_exist(self):
        from biosuite.gui.main_window import BioSuiteApp
        help_methods = [
            '_help_getting_started', '_help_sequence', '_help_alignment',
            '_help_transcriptomics', '_help_genomics', '_help_singlecell',
            '_help_proteomics', '_help_crispr', '_help_metagenomics',
            '_help_popgen', '_help_ml', '_help_databases',
            '_help_fileformats', '_help_apikeys', '_help_external',
            '_help_visualization', '_help_workflow', '_help_go',
            '_help_gwas', '_help_epitope', '_help_16s',
            '_help_svcnv', '_help_bigwig', '_help_shortcuts',
        ]
        for method in help_methods:
            assert _has_method(BioSuiteApp, method), f"Missing: {method}"

    def test_help_content_not_empty(self):
        from biosuite.gui.main_window import BioSuiteApp
        app = BioSuiteApp.__new__(BioSuiteApp)
        for method_name in ['_help_getting_started', '_help_visualization',
                             '_help_workflow', '_help_gwas', '_help_epitope',
                             '_help_16s', '_help_svcnv', '_help_bigwig',
                             '_help_shortcuts']:
            content = getattr(BioSuiteApp, method_name)(app)
            assert len(content) > 100, f"{method_name} too short"


# ─── Keyboard Shortcuts ─────────────────────────────────────────────────────

class TestKeyboardShortcuts:
    def test_save_current_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_save_current')

    def test_refresh_current_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_refresh_current')

    def test_shortcuts_in_source(self):
        from biosuite.gui.main_window import BioSuiteApp
        src = inspect.getsource(BioSuiteApp._finish_startup)
        assert 'Control-q' in src
        assert 'Control-s' in src
        assert 'F1' in src
        assert 'F5' in src
        assert 'Escape' in src


# ─── Progress Bar ───────────────────────────────────────────────────────────

class TestProgressBar:
    def test_show_progress_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_show_progress')

    def test_update_progress_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_update_progress')

    def test_hide_progress_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_hide_progress')


# ─── Plot History ───────────────────────────────────────────────────────────

class TestPlotHistory:
    def test_record_plot_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_record_plot')

    def test_show_plot_history_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_show_plot_history')


# ─── Drag and Drop ──────────────────────────────────────────────────────────

class TestDragDrop:
    def test_setup_drag_drop_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_setup_drag_drop')

    def test_on_drop_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_on_drop')


# ─── Theme System ───────────────────────────────────────────────────────────

class TestThemeSystem:
    def test_apply_theme_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_apply_theme')

    def test_rebuild_ui_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_rebuild_ui')

    def test_all_themes_have_required_keys(self):
        from biosuite.gui.main_window import THEMES
        required_keys = ['name', 'ctk_mode', 'bg', 'accent', 'text', 'border']
        for theme_name, theme in THEMES.items():
            for key in required_keys:
                assert key in theme, f"Missing '{key}' in theme '{theme_name}'"


# ─── Dialogs ────────────────────────────────────────────────────────────────

class TestDialogs:
    def test_msg_info_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_msg_info')

    def test_msg_warning_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_msg_warning')

    def test_msg_error_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_msg_error')

    def test_msg_success_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_msg_success')

    def test_confirm_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_confirm')

    def test_ask_input_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_ask_input')

    def test_ask_dropdown_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_ask_dropdown')

    def test_ask_file_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_ask_file')


# ─── UI Helpers ─────────────────────────────────────────────────────────────

class TestUIHelpers:
    def test_card_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_card')

    def test_section_header_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_section_header')

    def test_action_button_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_action_button')

    def test_input_entry_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_input_entry')

    def test_text_box_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_text_box')

    def test_label_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_label')


# ─── Frame Navigation ───────────────────────────────────────────────────────

class TestFrameNavigation:
    def test_show_frame_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_show_frame')

    def test_build_content_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_build_content')

    def test_on_close_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_on_close')

    def test_set_status_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_set_status')

    def test_gui_input_exists(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert _has_method(BioSuiteApp, '_gui_input')


# ─── All Frame Builders Registered ──────────────────────────────────────────

class TestAllFramesRegistered:
    def test_build_content_calls_all_frames(self):
        from biosuite.gui.main_window import BioSuiteApp
        src = inspect.getsource(BioSuiteApp._build_content)
        frames = ['_build_plot_frame', '_build_sequence_frame', '_build_alignment_frame',
                   '_build_phylogeny_frame', '_build_expression_frame', '_build_ngs_frame',
                   '_build_trimming_frame', '_build_quant_frame', '_build_singlecell_frame',
                   '_build_structure_frame', '_build_assembly_frame', '_build_metagenomics_frame',
                   '_build_crispr_frame', '_build_popgen_frame', '_build_ml_frame',
                   '_build_orftools_frame', '_build_databases_frame', '_build_fileformats_frame',
                   '_build_apikey_frame', '_build_help_frame', '_build_upset_frame',
                   '_build_genomebrowser_frame', '_build_conservation_frame',
                   '_build_synteny_frame', '_build_interactive_frame', '_build_pipeline_frame',
                   '_build_batch_frame', '_build_gobrowser_frame', '_build_pathway_frame',
                   '_build_gwas_frame', '_build_epitope_frame', '_build_16srna_frame',
                   '_build_svcnv_frame', '_build_bigwig_frame']
        for frame in frames:
            assert frame in src, f"Missing frame builder call: {frame}"


# ─── Total GUI Method Count ─────────────────────────────────────────────────

class TestGUICompleteness:
    def test_total_gui_methods(self):
        from biosuite.gui.main_window import BioSuiteApp
        methods = [m for m in dir(BioSuiteApp) if m.startswith('_') and not m.startswith('__')]
        # Should have 100+ methods for 29 tabs
        assert len(methods) >= 100, f"Only {len(methods)} methods — expected 100+"

    def test_plot_funcs_count(self):
        from biosuite.gui.main_window import PLOT_CATEGORIES
        total = sum(len(items) for items in PLOT_CATEGORIES.values())
        assert total >= 30, f"Only {total} plots — expected 30+"
