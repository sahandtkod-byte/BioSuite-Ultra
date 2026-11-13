"""The legend follows the caller's order, and it is the same order on every run.

Three functions built their colour-column groups with ``list(set(...))``.  The
trace order, the colour assignment and the legend were therefore whatever the
hash table happened to produce: the same six labels came out in two different
orders in two runs of the probe, so a figure the README promises is reproducible
was not.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip

GROUPS = ["wt", "mut", "ctrl", "rescue", "ko", "ts"]
N = len(GROUPS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_scatter_traces_follow_the_input_order():
    fig = ip.interactive_scatter(np.arange(N), np.arange(N), color_col=GROUPS)

    assert [t.name for t in fig.data] == GROUPS


def test_the_3d_traces_follow_the_input_order():
    fig = ip.interactive_3d_scatter(np.arange(N), np.arange(N), np.arange(N),
                                   color_col=GROUPS)

    assert [t.name for t in fig.data] == GROUPS


def test_the_pca_traces_follow_the_input_order():
    coords = np.column_stack([np.arange(N), np.arange(N)])

    fig = ip.interactive_pca(coords, color_col=GROUPS)

    assert [t.name for t in fig.data] == GROUPS


def test_the_fallback_legend_follows_the_input_order():
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_scatter(np.arange(N), np.arange(N), color_col=GROUPS)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    finally:
        ip.HAS_PLOTLY = True

    assert labels == GROUPS


def test_the_same_figure_is_drawn_identically_under_three_hash_seeds():
    """The proof the source cannot give: three fresh interpreters, one order."""
    code = (
        "from biosuite.plotting import interactive_plots as ip\n"
        "f = ip.interactive_scatter([0,1,2,3,4,5],[0,1,2,3,4,5],"
        "color_col=['wt','mut','ctrl','rescue','ko','ts'])\n"
        "print([t.name for t in f.data])\n"
    )
    orders = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONPATH=".;audit_output/.probe_env",
                   PYTHONHASHSEED=seed, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, env=env, timeout=120)
        assert result.returncode == 0, result.stderr[-400:]
        orders.add(result.stdout.strip())

    assert orders == {"['wt', 'mut', 'ctrl', 'rescue', 'ko', 'ts']"}, orders
