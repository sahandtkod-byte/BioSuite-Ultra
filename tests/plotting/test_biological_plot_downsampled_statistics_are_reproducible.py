"""The same table has to give the same plot and the same numbers.

``volcano_plot`` and ``pca_plot`` thinned big tables with an unseeded
``np.random.choice`` **before** reporting their statistics, so answering the same
questions over the same file printed a different answer every run — measured
"Total significant: 79", then 85, then 83 on one 6,000-row table.  Every other
plot in the package downsamples through ``utils.maybe_downsample``, which is an
even stride, keeps the endpoints and logs what it did.  The decomposition and the
counts now run on the whole table; only the points drawn are thinned.
"""
import contextlib
import io
import logging
import matplotlib
matplotlib.use('Agg')

import builtins

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.core.utils import config, session
from biosuite.plotting import biological_plots as bp

ROWS = 6000
LIMIT_KEY = 'downsample_threshold'


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
def big_table(tmp_path):
    path = tmp_path / 'de.csv'
    rng = np.random.default_rng(7)
    lfc = rng.normal(0, 1.5, ROWS)
    pvals = rng.uniform(0, 1, ROWS)
    pd.DataFrame({'log2fc': lfc, 'p': pvals}).to_csv(path, index=False)
    return str(path), lfc, pvals


@pytest.fixture
def run_volcano(monkeypatch):
    """Run the volcano demo over a table and report what it drew and logged."""
    def go(path):
        drawn = {}
        printed = io.StringIO()

        def spy(default_name, save_format, dpi, pdf=None, story=None):
            fig = plt.gcf()
            ax = fig.axes[0]
            drawn['points'] = sum(len(np.asarray(c.get_offsets()))
                                  for c in ax.collections)
            drawn['x'] = np.sort(np.concatenate(
                [np.asarray(c.get_offsets())[:, 0] for c in ax.collections])).tolist()
            plt.close(fig)
            return None

        answers = {"Load data from file": 'y', "File path": path,
                   "Log2FC column": 'log2fc', "P-value column": 'p'}
        table = {k.lower(): v for k, v in answers.items()}

        def ask(prompt=""):
            for key, value in table.items():
                if key in prompt.lower():
                    return value
            return ''

        monkeypatch.setattr(bp, 'ask_save_plot', spy)
        monkeypatch.setattr(builtins, 'input', ask)
        records = []

        class Catch(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        utils_logger = logging.getLogger('biosuite.core.utils')
        handler = Catch()
        utils_logger.addHandler(handler)
        try:
            with contextlib.redirect_stdout(printed):
                bp.volcano_plot()
        finally:
            utils_logger.removeHandler(handler)
        return {'drawn': drawn, 'log': records, 'stdout': printed.getvalue()}

    return go


def test_two_runs_of_the_same_file_draw_the_same_points(run_volcano, big_table):
    path, _lfc, _p = big_table

    first = run_volcano(path)
    second = run_volcano(path)

    assert first['drawn']['points'] == second['drawn']['points'] == config[LIMIT_KEY]
    assert first['drawn']['x'] == second['drawn']['x'], (
        "the same table drew different points: an unseeded np.random.choice")
    assert "simulated data set" not in first['stdout'], first['stdout']


def the_significant_count(messages):
    for message in messages:
        if 'Total significant:' in message:
            return int(message.split('Total significant:')[1])
    raise AssertionError(f"no significance line in {messages}")


def test_the_counted_genes_describe_the_whole_table_not_the_drawing(run_volcano, big_table):
    path, lfc, pvals = big_table

    report = run_volcano(path)

    expected = int(((np.abs(lfc) >= 1.0) & (pvals < 0.05)).sum())
    assert the_significant_count(report['log']) == expected, report['log']
    assert report['drawn']['points'] == config[LIMIT_KEY]
    assert report['drawn']['points'] < ROWS, "the drawing was meant to be thinned"


def test_the_thinning_is_announced(run_volcano, big_table):
    path, _lfc, _p = big_table

    report = run_volcano(path)

    assert any('Downsampled from 6000 to 5000' in message
               for message in report['log']), report['log']


def test_a_table_under_the_limit_is_drawn_whole(run_volcano, tmp_path):
    path = tmp_path / 'small.csv'
    pd.DataFrame({'log2fc': [2.0, -3.0, 0.1], 'p': [0.001, 0.02, 0.6]}).to_csv(
        path, index=False)

    report = run_volcano(str(path))

    assert report['drawn']['points'] == 3
    assert not any('Downsampled' in m for m in report['log']), report['log']


def test_pca_reports_the_variance_of_every_sample_and_draws_a_stride(
        monkeypatch, tmp_path):
    from sklearn.decomposition import PCA

    path = tmp_path / 'samples.csv'
    rng = np.random.default_rng(3)
    matrix = rng.normal(0, 1, (ROWS, 4))
    pd.DataFrame(matrix, columns=[f'g{i}' for i in range(4)]).to_csv(path, index=False)
    seen = {}

    def spy(default_name, save_format, dpi, pdf=None, story=None):
        fig = plt.gcf()
        seen['points'] = sum(len(np.asarray(c.get_offsets()))
                             for c in fig.axes[0].collections)
        seen['title'] = fig.axes[0].get_title()
        plt.close(fig)
        return None

    answers = {"Load data from file": 'y', "File path": str(path), "Group column": ''}
    table = {k.lower(): v for k, v in answers.items()}

    def ask(prompt=""):
        for key, value in table.items():
            if key in prompt.lower():
                return value
        return ''

    monkeypatch.setattr(bp, 'ask_save_plot', spy)
    monkeypatch.setattr(builtins, 'input', ask)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        bp.pca_plot()

    full = PCA(n_components=2).fit(matrix)
    expected = (f'PCA (var: {full.explained_variance_ratio_[0]:.2f}, '
                f'{full.explained_variance_ratio_[1]:.2f})')
    assert seen['title'] == expected, seen['title']
    assert seen['points'] == config[LIMIT_KEY]
    assert f"Drawing {config[LIMIT_KEY]} of {ROWS} samples" in stdout.getvalue(), \
        stdout.getvalue()
    assert "(the decomposition used all of them)" in stdout.getvalue()
