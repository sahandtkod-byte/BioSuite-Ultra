"""Regression tests for interactive specialized plots (input mocked)."""
import builtins
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest


def _with_inputs(*answers):
    """Wrap loop: monkeypatch builtins.input for each answer in order."""
    answers_iter = iter(list(answers))

    def fake_input(prompt=''):
        try:
            return next(answers_iter)
        except StopIteration:
            raise EOFError from None

    return fake_input


def test_gsea_plot_default_path(monkeypatch):
    from biosuite.plotting import specialized_plots as sp
    monkeypatch.setattr(builtins, 'input', _with_inputs('n'))
    sp.gsea_plot()
    plt.close('all')


def test_motif_logo_default_alignment(monkeypatch):
    from biosuite.plotting import specialized_plots as sp
    monkeypatch.setattr(builtins, 'input', _with_inputs('y'))
    sp.motif_logo()
    plt.close('all')


def test_sankey_diagram(monkeypatch):
    from biosuite.plotting import specialized_plots as sp
    monkeypatch.setattr(builtins, 'input', _with_inputs(''))
    sp.sankey_diagram()
    plt.close('all')


def test_umap_plot(monkeypatch):
    from biosuite.plotting import specialized_plots as sp
    monkeypatch.setattr(builtins, 'input', _with_inputs(''))
    sp.umap_plot()
    plt.close('all')
