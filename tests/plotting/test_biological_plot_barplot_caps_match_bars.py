"""Error caps belong on the bars they describe.

``barplot_custom`` drew the bars with ``sns.barplot``, which collapses repeated
categories into **one bar holding their mean**, and then added the caller's error
column with ``ax.errorbar(x=range(len(groups)), y=values, yerr=errors)`` — one cap
per *row*, at row positions.  Measured on ``A,A,B`` with three errors: two bars
came out (heights 2.0 and 6.0, the pair means) and the caps were placed for four
rows, so the picture showed an uncertainty that belongs to no bar.  seaborn's own
bootstrapped interval was drawn at the same time, in the same figure, describing
something else entirely.  Caps are now added only when one row really is one bar,
and a mistake in the error column is spoken out loud instead of drawn around.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import biological_plots as bp

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def bars(tmp_path):
    def write(rows, **answers):
        path = tmp_path / "bars.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    return write


def _answer_table(path, err_col="err"):
    table = {"Load data from file?": "y", "File path": path,
             "Category column": "cat", "Value column": "val",
             "Error column": err_col}
    return {key.lower(): value for key, value in table.items()}


def _caps(figure):
    """The collections ax.errorbar adds.  Measured: a barplot with no usable error
    column has zero collections, so anything here came from our own call."""
    return list(figure.axes[0].collections)


def _bars(figure):
    """Bar heights only — apply_glass_ax decorates the axes with a FancyBboxPatch."""
    return sorted(round(float(patch.get_height()), 2)
                  for patch in figure.axes[0].patches
                  if type(patch).__name__ == 'Rectangle')


def test_one_row_per_category_gets_one_cap_per_bar(run_demo, figure_at_save, bars):
    path = bars([{'cat': 'A', 'val': 1.0, 'err': 0.5},
                 {'cat': 'B', 'val': 3.0, 'err': 0.4},
                 {'cat': 'C', 'val': 5.0, 'err': 0.3}])
    captured = figure_at_save(bp)

    report = run_demo(bp.barplot_custom, answers=_answer_table(path))

    assert report["images"] == ["barplot.png"], report["stdout"]
    assert _caps(captured["fig"]), "the caller's error column drew nothing"
    assert _bars(captured["fig"]) == [1.0, 3.0, 5.0], _bars(captured["fig"])
    assert report["stdout"].count("not a number") == 0, report["stdout"]


def test_repeated_categories_are_collapsed_and_the_caps_are_withheld(
        run_demo, figure_at_save, bars):
    path = bars([{'cat': 'A', 'val': 1.0, 'err': 0.5},
                 {'cat': 'A', 'val': 3.0, 'err': 0.4},
                 {'cat': 'B', 'val': 5.0, 'err': 0.3}])
    captured = figure_at_save(bp)

    report = run_demo(bp.barplot_custom, answers=_answer_table(path))

    out = report["stdout"]
    assert "3 rows collapse into 2 categories" in out, out
    assert "the error column belongs to the rows and is not drawn" in out, out
    assert _caps(captured["fig"]) == [], "no cap may sit at a row position"
    # the pair means seaborn aggregated: (1.0 + 3.0) / 2, and 5.0
    assert _bars(captured["fig"]) == [2.0, 5.0], _bars(captured["fig"])
    assert [tick.get_text() for tick in captured["fig"].axes[0].get_xticklabels()] == ['A', 'B']


def test_an_error_column_that_is_not_in_the_table_is_named(run_demo, bars):
    path = bars([{'cat': 'A', 'val': 1.0, 'sd': 0.5}, {'cat': 'B', 'val': 3.0, 'sd': 0.4}])

    report = run_demo(bp.barplot_custom, answers=_answer_table(path, "sd"))

    # the column exists here, so this run must stay silent about it
    assert "is not a column of this table" not in report["stdout"], report["stdout"]

    report2 = run_demo(bp.barplot_custom, answers=_answer_table(path, "missing"))
    assert "'missing' is not a column of this table" in report2["stdout"], report2["stdout"]
    assert "without error caps" in report2["stdout"], report2["stdout"]


def test_blank_error_cells_are_counted(run_demo, bars):
    path = bars([{'cat': 'A', 'val': 1.0, 'err': 0.5},
                 {'cat': 'B', 'val': 3.0, 'err': np.nan},
                 {'cat': 'C', 'val': 5.0, 'err': 0.3}])

    report = run_demo(bp.barplot_custom, answers=_answer_table(path))

    out = report["stdout"]
    assert "1 of 3 values in 'err' are not a number" in out, out
    assert "those caps are absent" in out, out


def test_the_demo_bars_are_announced(run_demo, bars):
    path = bars([{'cat': 'A', 'val': 1.0}])

    report = run_demo(bp.barplot_custom,
                      answers={"load data from file": "y", "file path": path,
                               "category column": "nope", "value column": "val",
                               "error column": ""})

    assert "Drawing the built-in demo bars" in report["stdout"], report["stdout"]
    assert "not from your file" in report["stdout"], report["stdout"]
