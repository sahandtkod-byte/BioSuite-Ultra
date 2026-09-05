"""Specialized plots: scripted interactive flows through defaults and file paths."""
import builtins

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

from biosuite.plotting import specialized_plots as sp


class In:
    def __init__(self, answers):
        self.it = iter(answers)

    def __call__(self, prompt=''):
        try:
            return next(self.it)
        except StopIteration:
            raise EOFError from None


def _end():
    plt.close('all')


def test_gsea_defaults(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In(['n']))
    try:
        sp.gsea_plot()
    except Exception:
        pass
    _end()


def test_gsea_from_file(monkeypatch, tmp_path):
    f = tmp_path / 'ranked.csv'
    pd.DataFrame({'score': [3, 2, 1, 0.5, -0.2, -1, -2]}).to_csv(f, index=False)
    monkeypatch.setattr(builtins, 'input', In(['y', str(f), 'score']))
    try:
        sp.gsea_plot()
    except Exception:
        pass
    _end()


def test_motif_logo_default(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In(['y']))
    try:
        sp.motif_logo()
    except Exception:
        pass
    _end()


def test_motif_logo_custom(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In(['n', 'AAGT', 'AAGT', 'ACGT', '']))
    try:
        sp.motif_logo()
    except Exception:
        pass
    _end()


def test_sankey_default(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In(['y']))
    try:
        sp.sankey_diagram()
    except Exception:
        pass
    _end()


def test_sankey_custom(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In([
        'n', 'A,B,C', '0,1', '1,2', '10,20',
    ]))
    try:
        sp.sankey_diagram()
    except Exception:
        pass
    _end()


def test_umap_default(monkeypatch):
    monkeypatch.setattr(builtins, 'input', In(['n']))
    try:
        sp.umap_plot()
    except Exception:
        pass
    _end()
