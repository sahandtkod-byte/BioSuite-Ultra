"""A correlation that cannot be computed is not a correlation of zero.

``heatmap_custom`` colours ``numeric_df.corr()`` — and pandas answers a column
that never varies with ``NaN`` against everything, including itself.  seaborn
paints NaN as a blank cell, so a table with one constant column produced a matrix
with a blank row and a blank column that reads, to everyone looking at it, as
"no association".  Nothing distinguished undefined from zero, and nothing named
the column that caused it.
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
def matrix(tmp_path):
    def write(**columns):
        path = tmp_path / "corr.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        return str(path)

    return write


def answers_for(path):
    return {"load data from file": "y", "file path": path}


def test_a_usable_table_is_drawn_without_an_apology(run_demo, matrix):
    path = matrix(x=[1.0, 2.0, 3.0], y=[2.0, 4.0, 6.0], z=[3.0, 1.0, 2.0])

    report = run_demo(bp.heatmap_custom, answers=answers_for(path))

    out = report["stdout"]
    assert "no correlation" not in out, out
    assert report["images"] == ["heatmap.png"], out
    assert report["figure_title"] == "Correlation Heatmap", report


def test_a_constant_column_is_named_and_the_blank_cells_are_explained(run_demo, matrix):
    path = matrix(x=[1.0, 2.0, 3.0], y=[2.0, 4.0, 6.0], z=[1.0, 1.0, 1.0])

    report = run_demo(bp.heatmap_custom, answers=answers_for(path))

    out = report["stdout"]
    assert "2 column pair(s) have no correlation" in out, out
    assert "a column never varies: z." in out, out
    assert "Those cells are blank — undefined, not zero." in out, out
    assert report["images"] == ["heatmap.png"], out


def test_two_constant_columns_are_both_counted(run_demo, matrix):
    path = matrix(a=[1.0, 1.0], b=[2.0, 2.0], c=[1.0, 3.0])

    report = run_demo(bp.heatmap_custom, answers=answers_for(path))

    out = report["stdout"]
    assert "3 column pair(s) have no correlation" in out, out
    assert "a column never varies: a, b." in out, out
    assert "never varies: a, b, c" not in out, out


def test_a_single_constant_column_is_named_even_without_any_pair(run_demo, matrix):
    path = matrix(only=[1.0, 1.0, 1.0])

    report = run_demo(bp.heatmap_custom, answers=answers_for(path))

    out = report["stdout"]
    assert "0 column pair(s) have no correlation because a column never varies: only" \
        in out, out


def test_a_fully_constant_column_pair_is_not_reported_when_it_is_defined(
        run_demo, matrix):
    path = matrix(x=[1.0, 2.0], y=[1.0, 2.0])

    report = run_demo(bp.heatmap_custom, answers=answers_for(path))

    assert "no correlation because" not in report["stdout"], report["stdout"]
