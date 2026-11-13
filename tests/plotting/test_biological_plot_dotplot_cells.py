"""Single-cell dot plots: one cell per gene and cluster, and one scale for the
percent column.

``dot_plot`` reshaped the table with ``df.pivot(...)``, which raises
``ValueError: Index contains duplicate entries, cannot reshape`` the moment a
marker appears twice in one cluster — the ordinary shape of a table with
replicates.  The demo's handler turned that into one ``Error:`` line and drew
nothing.  And the marker area is computed as ``20 + pct * 80``, which is only
sensible for a 0-1 fraction: measured on a column of ``80/90/100`` the first
marker came out at **6420 points squared** against 84 for the same data written
as fractions, so the plot was a stack of balloons.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import biological_plots as bp

ANSWERS = {"gene column": "gene", "cluster column": "cluster",
           "percent expressed column": "pct", "average expression column": "exp"}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def dot_table(tmp_path):
    def write(rows):
        path = tmp_path / "dot.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    return write


# One marker is one ax.scatter call, so the figure holds one PathCollection per
# point: read every collection, not just the first.
def _sizes(fig):
    return [int(size) for collection in fig.axes[0].collections
            for size in np.atleast_1d(collection.get_sizes())]


def _positions(fig):
    return [[float(x), float(y)] for collection in fig.axes[0].collections
            for x, y in np.asarray(collection.get_offsets())]


def test_one_row_per_cell_still_draws_every_marker(run_demo, figure_at_save, dot_table):
    path = dot_table([{'gene': g, 'cluster': 'c1', 'pct': p, 'exp': e}
                      for g, p, e in [('A', 0.8, 1.0), ('B', 0.9, 2.0),
                                      ('C', 1.0, 3.0)]])

    captured = figure_at_save(bp)
    report = run_demo(bp.dot_plot, answers={"load data from file": "y",
                                            "file path": path, **ANSWERS})

    assert report["images"] == ["dotplot.png"], report["stdout"]
    assert "duplicate" not in report["stdout"], report["stdout"]
    assert _sizes(captured["fig"]) == [84, 92, 100]
    assert _positions(captured["fig"]) == [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]


def test_a_percent_column_is_rescaled_and_says_so(run_demo, figure_at_save, dot_table):
    path = dot_table([{'gene': g, 'cluster': 'c1', 'pct': p, 'exp': e}
                      for g, p, e in [('A', 80.0, 1.0), ('B', 90.0, 2.0),
                                      ('C', 100.0, 3.0)]])

    captured = figure_at_save(bp)
    report = run_demo(bp.dot_plot, answers={"load data from file": "y",
                                            "file path": path, **ANSWERS})

    assert "on a 0-100 scale" in report["stdout"], report["stdout"]
    assert "divided by 100" in report["stdout"], report["stdout"]
    assert _sizes(captured["fig"]) == [84, 92, 100], "the same markers as the fraction"
    assert report["images"] == ["dotplot.png"], report["stdout"]


def test_replicate_rows_are_averaged_and_counted(run_demo, figure_at_save, dot_table):
    path = dot_table([
        {'gene': 'A', 'cluster': 'c1', 'pct': 0.4, 'exp': 1.0},
        {'gene': 'A', 'cluster': 'c1', 'pct': 0.6, 'exp': 3.0},
        {'gene': 'B', 'cluster': 'c2', 'pct': 0.2, 'exp': 2.0},
    ])

    captured = figure_at_save(bp)
    report = run_demo(bp.dot_plot, answers={"load data from file": "y",
                                            "file path": path, **ANSWERS})

    out = report["stdout"]
    assert "1 gene/cluster cell(s) hold more than one row; they are averaged" in out, out
    assert "Error:" not in out, out
    assert report["images"] == ["dotplot.png"], out
    # A(c1) is the mean of the two rows: pct 0.5 -> area 60, exp 2.0
    assert _sizes(captured["fig"])[0] == 60, _sizes(captured["fig"])


def test_a_duplicated_cell_does_not_leave_the_demo_without_a_plot(
        run_demo, dot_table):
    path = dot_table([{'gene': 'A', 'cluster': 'c1', 'pct': 0.5, 'exp': 1.0},
                      {'gene': 'A', 'cluster': 'c1', 'pct': 0.5, 'exp': 2.0}])

    report = run_demo(bp.dot_plot, answers={"load data from file": "y",
                                            "file path": path, **ANSWERS})

    assert report["figures_open"] == 0, report["stdout"]
    assert "Error:" not in report["stdout"], report["stdout"]


def test_a_missing_column_still_announces_the_demo_data(run_demo, dot_table, figure_at_save):
    path = dot_table([{'gene': 'A', 'cluster': 'c1', 'pct': 0.5, 'exp': 1.0}])

    report = run_demo(bp.dot_plot, answers={
        "load data from file": "y", "file path": path,
        "gene column": "marker", "cluster column": "cluster",
        "percent expressed column": "pct", "average expression column": "exp"})

    assert "Using default demo data." in report["stdout"], report["stdout"]
    assert report["images"] == ["dotplot.png"], report["stdout"]
