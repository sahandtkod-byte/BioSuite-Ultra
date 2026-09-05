"""Regression tests for export completeness + polar glass styling."""
import matplotlib
matplotlib.use('Agg')
import os
import tempfile

import matplotlib.pyplot as plt
import pytest


def test_export_all_to_folder_writes_math_and_specialized(tmp_path):
    """Regression: only biological-plots were monkeypatched before —
    sine/cosine/logistic/GSEA/sankey/... were silently never written."""
    from biosuite.plotting.biological_plots import export_all_to_folder
    out = str(tmp_path / 'export')
    export_all_to_folder(out)
    files = set(os.listdir(out))
    for expected in ('volcano.png', 'sine.png', 'logistic.png',
                     'gsea.png', 'sankey.png', 'dotplot.png'):
        assert expected in files, f"missing: {expected}"


def test_export_restore_state(tmp_path):
    """ask_save_plot must be restored on all three modules afterwards."""
    from biosuite.plotting import biological_plots, math_plots, specialized_plots
    b_orig = biological_plots.ask_save_plot
    m_orig = math_plots.ask_save_plot
    out = str(tmp_path / 'export2')
    biological_plots.export_all_to_folder(out)
    assert biological_plots.ask_save_plot is b_orig
    assert math_plots.ask_save_plot is m_orig
    assert specialized_plots.ask_save_plot is m_orig


def test_apply_glass_ax_tolerates_polar():
    from biosuite.core.utils import apply_glass_ax
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    apply_glass_ax(ax)            # previously KeyError('top')
    plt.close('all')


def test_circos_renders_after_glass_fix():
    from biosuite.plotting.biological_plots import circos_plot
    import builtins
    orig_input = builtins.input
    builtins.input = lambda prompt='': 'y'   # use default data
    try:
        circos_plot(pdf=None)
    finally:
        builtins.input = orig_input
    plt.close('all')
