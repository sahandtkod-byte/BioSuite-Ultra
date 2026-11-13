"""An UpSet plot drops sets before it counts their combinations.

The table doubles with every set, so a 12-set request for the largest 5 used to
build 4,095 rows before throwing 7 sets away.  The plot now trims first, and the
title records how many of how many were kept.  Its spacing comes from the
gridspec, not from a tight-layout pass that cannot touch a locally-set hspace.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

import biosuite.plotting.upset_plots as upset_plots


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _twelve_sets():
    sets = {f"S{i}": {i, 100 + i} for i in range(12)}
    sets["S0"] = {0, 1, 2, 3, 4, 5}
    sets["S1"] = {1, 2, 3, 4}
    sets["S2"] = {2, 3}
    return sets


def test_the_matrix_is_computed_from_the_sets_that_are_kept(monkeypatch):
    seen = []
    original = upset_plots.compute_upset_matrix

    def spy(sets_dict):
        seen.append(len(sets_dict))
        return original(sets_dict)

    monkeypatch.setattr(upset_plots, "compute_upset_matrix", spy)

    fig = upset_plots.plot_upset(_twelve_sets(), max_sets=5)

    assert seen == [5], f"the matrix was computed for {seen} sets"
    plt.close(fig)


def test_the_title_records_the_sets_that_were_dropped():
    fig = upset_plots.plot_upset(_twelve_sets(), max_sets=5, title="Shared genes")

    assert fig.axes[0].get_title() == "Shared genes (top 5 of 12 sets)"


def test_a_table_that_fits_keeps_the_callers_title():
    fig = upset_plots.plot_upset({"A": {1, 2}, "B": {2, 3}}, title="Shared genes")

    assert fig.axes[0].get_title() == "Shared genes"


def test_the_plot_does_not_ask_tight_layout_to_redo_its_spacing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = upset_plots.plot_upset({"A": {1, 2}, "B": {2, 3}})
        fig.canvas.draw()

    tight = [w for w in caught if "tight_layout" in str(w.message)]
    assert tight == [], f"the UpSet plot warned about its own layout: {tight}"
