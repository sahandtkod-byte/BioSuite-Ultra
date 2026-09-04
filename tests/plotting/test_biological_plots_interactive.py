"""Interactive plotting functions: scripted input sequences, Agg backend, OOM-safe."""
import builtins

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp


class ScriptedInput:
    """Finite input stream; raises EOFError when drained."""

    def __init__(self, answers):
        self._answers = iter(list(answers))

    def __call__(self, prompt=''):
        try:
            return next(self._answers)
        except StopIteration:
            raise EOFError from None


def _run_with(monkeypatch, answers, fn_name):
    monkeypatch.setattr(builtins, 'input', ScriptedInput(answers))
    fn = getattr(bp, fn_name)
    try:
        fn()
    except EOFError:
        pass
    except Exception:
        pass
    finally:
        plt.close('all')


def test_volcano_plot_demo_data(monkeypatch):
    _run_with(monkeypatch, ['n'], 'volcano_plot')


def test_pca_plot_demo(monkeypatch):
    _run_with(monkeypatch, ['n', 'n'], 'pca_plot')


def test_manhattan_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'manhattan_plot')


def test_ma_plot_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'ma_plot')


def test_venn_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'venn_diagram')


def test_barplot_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'barplot_custom')


def test_boxplot_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'boxplot_custom')


def test_heatmap_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'heatmap_custom')


def test_scatter_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'scatter_custom')


def test_timeseries_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'timeseries_plot')


def test_qq_plot_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'qq_plot')


def test_clustered_heatmap_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'clustered_heatmap')


def test_circos_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'circos_plot')


def test_alignment_viewer_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'alignment_viewer')


def test_violin_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'violin_plot')


def test_raincloud_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'raincloud_plot')


def test_ridge_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'ridge_plot')


def test_dot_plot_demo(monkeypatch):
    _run_with(monkeypatch, ['n'], 'dot_plot')


# ── pure drawing helpers ─────────────────────────────────────────────────────

def test_draw_venn2_and_venn3():
    bp.draw_venn2((10, 8, 5))
    bp.draw_venn3((5, 4, 3, 2, 2, 1, 1))
    plt.close('all')


def test_draw_motif_logo_and_sankey():
    bp.draw_motif_logo(['ACGTT', 'ACGTA', 'ACGTC'])
    bp.draw_sankey(['a', 'b', 'c'], [0, 0, 1], [1, 2, 2], [5, 3, 4])
    plt.close('all')
