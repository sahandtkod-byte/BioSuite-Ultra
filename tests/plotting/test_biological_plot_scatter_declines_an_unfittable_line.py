"""The scatter demo declines a line it cannot fit, and says why.

``scatter_custom`` drew the figure first and only then asked scipy for the fit, so
a table whose x column never moved raised ``ValueError: Cannot calculate a linear
regression if all x values are identical`` inside the demo's handler: the user got
one ``Error:`` line, an empty figure left open for the life of the process, and no
points — the data that *was* valid was lost with the fit that was not.  A file
with a single usable row fell back to the simulated 20 points without a word, and
Spearman's rho answered a one-valued column with NaN printed as a statistic.
"""
import matplotlib
matplotlib.use('Agg')

import builtins
import contextlib
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.core.utils import config, session
from biosuite.plotting import biological_plots as bp


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
def drive(monkeypatch, tmp_path):
    """Run scatter_custom over an x/y table, returning what it drew and printed."""
    def go(x, y, corr_type='pearson'):
        path = tmp_path / "xy.csv"
        pd.DataFrame({"x": x, "y": y}).to_csv(path, index=False)
        captured = {}

        def spy(default_name, save_format, dpi, pdf=None, story=None):
            fig = plt.gcf()
            ax = fig.axes[0]
            captured['title'] = ax.get_title()
            captured['points'] = sum(len(np.asarray(collection.get_offsets()))
                                      for collection in ax.collections)
            captured['lines'] = len(ax.lines)
            captured['texts'] = [t.get_text() for t in ax.texts]
            plt.close(fig)
            return None

        answers = {"load data from file": 'y', "file path": str(path),
                   "correlation type": corr_type, "x column": 'x', "y column": 'y'}
        table = {k.lower(): v for k, v in answers.items()}

        def ask(prompt=""):
            for key, value in table.items():
                if key in prompt.lower():
                    return value
            return ''

        monkeypatch.setattr(bp, 'ask_save_plot', spy)
        monkeypatch.setattr(builtins, 'input', ask)
        printed = io.StringIO()
        figures_before = set(plt.get_fignums())
        with contextlib.redirect_stdout(printed):
            bp.scatter_custom()
        return captured, printed.getvalue(), set(plt.get_fignums()) - figures_before

    return go


def test_identical_x_values_still_get_their_points_and_an_explanation(drive):
    captured, out, leaked = drive([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

    assert "Every x value is 2.0, so no line can be fitted" in out, out
    assert "the points are drawn without one" in out, out
    assert captured['title'] == 'Scatter (no line fitted)', captured
    assert captured['points'] == 3, "the user's own points survived the refused fit"
    assert captured['lines'] == 0, "no line is claimed"
    assert captured['texts'] == [], "no R-squared is printed for a fit that failed"
    assert leaked == set(), f"the refused fit left {leaked} open"


def test_a_constant_y_column_refuses_spearman_instead_of_printing_nan(drive):
    captured, out, leaked = drive([1.0, 2.0, 3.0], [5.0, 5.0, 5.0],
                                  corr_type='spearman')

    assert "Spearman's rho needs two distinct values in both columns" in out, out
    assert "nan" not in "".join(captured['texts']).lower(), captured['texts']
    assert captured['points'] == 3
    assert leaked == set()


def test_a_one_row_table_is_not_mistaken_for_a_fit(drive):
    captured, out, leaked = drive([1.0], [2.0])

    assert "gives fewer than two usable 'x'/'y' pairs" in out, out
    assert "Drawing the built-in simulated points" in out, out
    assert captured['points'] == 20, "the demo's own 20 points, announced as such"
    assert leaked == set()


def test_a_fittable_table_still_annotates_its_line(drive):
    captured, out, leaked = drive([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

    assert "no line can be fitted" not in out, out
    assert captured['title'] == 'Scatter with Regression', captured
    assert captured['points'] == 3
    assert captured['lines'] == 1, "the fitted line"
    assert any(text.startswith('y = 2.000x + 0.000') for text in captured['texts']), \
        captured['texts']
    assert leaked == set(), "a saved demo releases its figure"
