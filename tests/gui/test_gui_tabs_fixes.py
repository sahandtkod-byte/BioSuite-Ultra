"""Structural regression tests for GUI tab fixes (attributes + display plumbing)."""
import inspect


def _src(mod_path):
    import importlib
    return inspect.getsource(importlib.import_module(mod_path))


def test_genomics_metagenomics_attrs_no_collision():
    g = _src('biosuite.gui.tabs.genomics')
    m = _src('biosuite.gui.tabs.metabolomics')
    # genomics uses mg_* names so the metabolomics tab's meta_path/meta_result
    # (same parent bio-suite instance) stay untouched.
    assert 'self.mg_path' in g and 'self.mg_result' in g
    fn = inspect.getsource(__import__('biosuite.gui.tabs.genomics', fromlist=['x'])
                           .GenomicsTabMixin._run_meta)
    assert 'self.mg_path' in fn and 'self.meta_path' not in fn
    assert 'self.meta_path' in m and 'self.mg_path' not in m


def test_genomics_load_csv_runs_analysis():
    src = inspect.getsource(__import__('biosuite.gui.tabs.genomics', fromlist=['x'])
                            .GenomicsTabMixin._svcnv_load)
    assert 'detect_structural_variants' in src
    assert 'detect_cnv' in src


def test_pathway_figures_are_displayed():
    mod = __import__('biosuite.gui.tabs.workflow', fromlist=['x'])
    for fn_name in ('_draw_pathway', '_kegg_demo'):
        src = inspect.getsource(getattr(mod.WorkflowTabMixin, fn_name))
        assert '_show_plot_from_figure' in src, fn_name
        assert '_record_plot' in src, fn_name


def test_gwas_load_csv_runs_analysis():
    src = inspect.getsource(__import__('biosuite.gui.tabs.workflow', fromlist=['x'])
                            .WorkflowTabMixin._gwas_load)
    assert 'run_gwas' in src and 'format_gwas_report' in src
