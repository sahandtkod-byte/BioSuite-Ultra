"""A demo that draws its own data has to say that is what it is doing.

Every plot demo in ``biological_plots.py`` falls back to a built-in simulated
data set when the file it was handed cannot be used — a column name typed wrong,
a table with one numeric column, fewer than two usable pairs.  That fallback is
deliberate (the input-branch tests pin the plot that arrives), but in nine of the
fourteen places the demo printed nothing at all: the user answered a "File path:"
prompt, named their own experiment, and got 500 random genes under a title that
said *Volcano Plot*.  ``heatmap_custom``, ``timeseries_plot``, ``dot_plot`` and
``ridge_plot`` already told the truth about the same situation, so the sentence
is not new UI — it is the file being consistent with itself.
"""
import contextlib
import io
import matplotlib
matplotlib.use('Agg')

import builtins

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.core.utils import config, session
from biosuite.plotting import biological_plots as bp

TWO_COLUMN = {'log2fc': [2.0, -3.0, 0.1], 'p': [0.001, 0.02, 0.6]}
GROUPED = {'group': ['ctrl', 'ctrl', 'trt', 'trt'], 'value': [1.0, 2.0, 5.0, 6.0]}


@pytest.fixture(autouse=True)
def _clean_state():
    saved = dict(session)
    config['quiet'] = True
    yield
    session.clear()
    session.update(saved)
    config.pop('quiet', None)
    plt.close('all')


@pytest.fixture
def csv_of(tmp_path):
    def write(name, data, index=False):
        path = tmp_path / name
        pd.DataFrame(data).to_csv(path, index=index)
        return str(path)

    return write


@pytest.fixture
def captured_figure(monkeypatch):
    """The figure each demo is about to save, plus everything it printed."""
    def watch(func, answers):
        seen = {}

        def spy(default_name, save_format, dpi, pdf=None, story=None):
            fig = plt.gcf()
            seen['name'] = default_name
            seen['fig'] = fig
            seen['texts'] = [t.get_text() for t in fig.axes[0].texts] if fig.axes else []
            seen['points'] = (sum(len(np.asarray(c.get_offsets()))
                                  for c in fig.axes[0].collections)
                              if fig.axes else 0)
            seen['title'] = fig.axes[0].get_title() if fig.axes else ''
            plt.close(fig)
            return None

        real = bp.ask_save_plot
        table = {key.lower(): value for key, value in answers.items()}

        def ask(prompt=""):
            for key, value in table.items():
                if key in prompt.lower():
                    return value
            return ''

        buffer = io.StringIO()
        monkeypatch.setattr(bp, 'ask_save_plot', spy)
        monkeypatch.setattr(builtins, 'input', ask)
        with contextlib.redirect_stdout(buffer):
            func()
        monkeypatch.setattr(bp, 'ask_save_plot', real)
        return {'seen': seen, 'stdout': buffer.getvalue()}

    return watch


def test_a_column_that_is_not_in_the_file_names_the_columns_that_are(
        captured_figure, csv_of):
    path = csv_of('genes.csv', TWO_COLUMN)

    report = captured_figure(bp.volcano_plot, {
        "Load data from file": "y", "File path": path,
        "Log2FC column": "foldchange", "P-value column": "p"})

    out = report['stdout']
    assert "no column(s) 'foldchange'" in out, out
    assert "its columns are log2fc, p." in out, out
    assert "Drawing the built-in simulated data set (500 genes) instead" in out, out
    assert "not from your file" in out, out
    assert "Error:" not in out, out
    assert report['seen']['points'] == 500


def test_a_file_whose_rows_are_not_numeric_is_said_so(captured_figure, csv_of):
    path = csv_of('text.csv', {'log2fc': ['high', 'low'], 'p': ['nsig', 'sig']})

    report = captured_figure(bp.volcano_plot, {
        "Load data from file": "y", "File path": path,
        "Log2FC column": "log2fc", "P-value column": "p"})

    out = report['stdout']
    assert "has no numeric rows in" in out, out
    assert "Drawing the built-in simulated data set" in out, out


def test_a_table_with_one_numeric_column_is_refused_out_loud(captured_figure, csv_of):
    path = csv_of('one.csv', {'g': ['a', 'b', 'c', 'd'], 'v': [1.0, 2.0, 3.0, 4.0]})

    report = captured_figure(bp.pca_plot, {"Load data from file": "y", "File path": path})

    out = report['stdout']
    assert "1 numeric column(s); PCA needs at least two features" in out, out
    assert "Drawing the built-in simulated data set" in out, out
    assert "Error:" not in out, out


def test_a_single_sample_table_is_named_as_too_small_for_pca(captured_figure, csv_of):
    path = csv_of('one_row.csv', {'v1': [1.0], 'v2': [2.0]})

    report = captured_figure(bp.pca_plot, {"Load data from file": "y", "File path": path})

    assert "1 sample row(s); PCA needs at least two samples" in report['stdout']


def test_an_unknown_group_column_says_which_groups_it_used(captured_figure, csv_of):
    path = csv_of('samples.csv', {**{f'g{i}': [1.0, 2.0, 3.0, 4.0] for i in range(3)},
                                  'genotype': ['a', 'a', 'b', 'b']})

    report = captured_figure(bp.pca_plot, {
        "Load data from file": "y", "File path": path, "Group column": "nope"})

    out = report['stdout']
    assert "'nope' is not a column of this table" in out, out
    assert "one group called 'Sample'" in out, out
    labels = {t.get_text() for t in report['seen']['fig'].axes[0].get_legend().get_texts()}
    assert labels == {'Sample'}, labels


def test_the_group_demos_announce_their_own_data(captured_figure, csv_of):
    path = csv_of('counts.csv', GROUPED)
    cases = [(bp.boxplot_custom, 'boxplot', 'Drawing the built-in demo groups'),
             (bp.violin_plot, 'violin', 'Drawing the built-in demo groups'),
             (bp.raincloud_plot, 'raincloud', 'Drawing the built-in demo groups')]

    for func, image, sentence in cases:
        report = captured_figure(func, {
            "Load data from file": "y", "File path": path,
            "Value column": "abundance", "Group column": "genotype"})
        out = report['stdout']
        assert sentence in out, f"{func.__name__}: {out}"
        assert "not your file" in out, func.__name__
        assert "Error:" not in out, func.__name__
        assert report['seen']['name'] == image, func.__name__


def test_scatter_announces_a_file_it_cannot_use(captured_figure, csv_of):
    path = csv_of('xy.csv', {'a': [1.0, 2.0], 'b': [3.0, 4.0]})

    report = captured_figure(bp.scatter_custom, {
        "Load data from file": "y", "Correlation type": "pearson", "File path": path,
        "X column": "dose", "Y column": "b"})

    out = report['stdout']
    assert "no column(s) 'dose'" in out, out
    assert "Drawing the built-in simulated points" in out, out


def test_qqplot_announces_a_column_it_cannot_read(captured_figure, csv_of):
    path = csv_of('vals.csv', {'x': ['a', 'b', 'c']})

    report = captured_figure(bp.qq_plot, {
        "Load data from file": "y", "File path": path, "Column to test": "x"})

    out = report['stdout']
    assert "200 simulated standard-normal draws" in out, out
    assert "not your data" in out, out


def test_the_barplot_announces_its_demo_bars(captured_figure, csv_of):
    path = csv_of('bars.csv', GROUPED)

    report = captured_figure(bp.barplot_custom, {
        "Load data from file": "y", "File path": path,
        "Category column": "genotype", "Value column": "missing",
        "Error column": ""})

    out = report['stdout']
    assert "Drawing the built-in demo bars" in out, out
    assert report['seen']['name'] == 'barplot', report['seen']['name']


def test_ridge_announces_the_demo_groups_it_fell_back_to(captured_figure, csv_of):
    path = csv_of('ridge.csv', GROUPED)

    report = captured_figure(bp.ridge_plot, {
        "Load data from file": "y", "File path": path,
        "Value column": "nope", "Group column": "group"})

    out = report['stdout']
    assert "Drawing the built-in demo ridges" in out, out
    assert "this is not your file" in out, out
    assert "Error:" not in out, out
