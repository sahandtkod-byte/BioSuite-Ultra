"""When ``interactive=True`` cannot be honoured, the caller has to hear about it.

Every plot in ``plot_api`` answers a request for an interactive figure with a
matplotlib one when Plotly is not installed.  That degradation is deliberate and
documented — what was missing was any sentence saying it happened, so
``fig.write_html("volcano.html")`` died with ``AttributeError: 'Figure' object
has no attribute 'write_html'`` and the GUI text above these functions went on
promising "Interactive plots are exported as HTML files".
"""
import matplotlib
matplotlib.use('Agg')

import inspect
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

LOG2FC = np.array([2.5, -3.0, 0.1])
PVALUES = np.array([0.001, 0.002, 0.60])

#: The functions whose signature takes ``interactive``, read off the module itself.
INTERACTIVE_PLOTS = sorted(
    name for name, value in vars(pa).items()
    if not name.startswith('_') and callable(value)
    and 'interactive' in inspect.signature(value).parameters)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _fallback_records(caplog):
    return [r for r in caplog.records
            if 'Plotly is not installed' in r.getMessage()]


def test_the_substitution_is_logged_by_the_module_that_made_it(monkeypatch, caplog):
    monkeypatch.setattr(pa, 'HAS_PLOTLY', False)

    with caplog.at_level('WARNING'):
        fig = pa.volcano(LOG2FC, PVALUES, interactive=True)

    assert isinstance(fig, plt.Figure), "the documented fallback figure still arrives"
    records = _fallback_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].name == 'biosuite.plotting.plot_api'
    message = records[0].getMessage()
    assert 'no write_html()' in message and 'output_html=' in message


def test_a_static_call_says_nothing_about_plotly(monkeypatch, caplog):
    """Positive control: the warning tracks the request, not the backend."""
    monkeypatch.setattr(pa, 'HAS_PLOTLY', False)

    with caplog.at_level('WARNING'):
        pa.volcano(LOG2FC, PVALUES)

    assert _fallback_records(caplog) == []


def test_a_call_that_plotly_can_answer_says_nothing(monkeypatch, caplog):
    pytest.importorskip('plotly.graph_objects')
    monkeypatch.setattr(pa, 'HAS_PLOTLY', True)

    with caplog.at_level('WARNING'):
        fig = pa.volcano(LOG2FC, PVALUES, interactive=True)

    assert _fallback_records(caplog) == []
    assert not isinstance(fig, plt.Figure)


def test_every_plot_that_takes_interactive_routes_through_the_guard():
    """A structural pin: the fallback message exists once, so a new plot cannot
    re-add the silent version by writing the condition by hand."""
    source = pathlib.Path(pa.__file__).read_text(encoding='utf-8')

    assert INTERACTIVE_PLOTS == [
        'barplot', 'boxplot', 'heatmap', 'ma', 'manhattan', 'pca', 'qqplot',
        'scatter', 'timeseries', 'venn', 'violin', 'volcano']
    assert source.count('if interactive and _plotly_available(interactive):') == 12
    assert 'if interactive and HAS_PLOTLY:' not in source


def test_the_guard_fires_once_per_plot_call(monkeypatch, caplog):
    monkeypatch.setattr(pa, 'HAS_PLOTLY', False)

    with caplog.at_level('WARNING'):
        pa.pca(np.random.default_rng(0).normal(size=(6, 3)), interactive=True)
        pa.qqplot(PVALUES, interactive=True)

    assert len(_fallback_records(caplog)) == 2
