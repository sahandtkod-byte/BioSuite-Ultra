"""A dotplot says when it has to skip the marginal histograms.

An external axes has no neighbouring axes to fill, so ``show_histograms`` cannot
be honoured there.  The old code skipped them silently; the note names the
request that was dropped, while a dotplot that builds its own figure stays
quiet and gets its three panels.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.synteny import plot_dotplot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_histogram_request_on_a_supplied_axes_is_reported(capsys):
    outer, ax = plt.subplots()

    fig = plot_dotplot("ACGTACGT", "ACGTACGT", word_size=3,
                       ax=ax, show_histograms=True)

    assert fig is outer
    assert len(outer.axes) == 1, "no marginal axes can appear beside the caller's axes"
    note = capsys.readouterr().out
    assert "show_histograms" in note
    assert "ignored" in note


def test_a_histogram_request_on_its_own_figure_does_not_need_a_note(capsys):
    fig = plot_dotplot("ACGTACGT", "ACGTACGT", word_size=3, show_histograms=True)

    assert len(fig.axes) == 3
    assert capsys.readouterr().out == ""


def test_a_supplied_axes_without_a_histogram_request_stays_quiet(capsys):
    outer, ax = plt.subplots()

    fig = plot_dotplot("ACGTACGT", "ACGTACGT", word_size=3, ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1
    assert capsys.readouterr().out == ""
