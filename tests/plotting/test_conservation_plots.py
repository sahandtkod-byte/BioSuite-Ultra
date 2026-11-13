"""The conservation scores, the logo, and the bar drawn under it.

The three original pins moved here in cluster 31; cluster 32 grew the file with
the bar panel and the combined layout, so the two entry points are checked
against the same scores.
"""
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.conservation_plots import (
    compute_conservation_scores,
    compute_logo_heights,
    plot_conservation_bar,
    plot_logo_with_conservation,
    plot_sequence_logo,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_logo_conserved_column_schneider_correction():
    # 4 identical sequences, single base -> e_n = 3/(2*ln2*4) = 0.541
    _, _, tot = compute_logo_heights(["AAA", "AAA", "AAA", "AAA"])
    assert tot[0] == pytest.approx(2 - 3 / (2 * math.log(2) * 4), abs=1e-9)


def test_logo_uniform_is_zero_bits():
    _, _, tot = compute_logo_heights(["AAA", "CCC", "GGG", "TTT"])
    assert all(t == 0 for t in tot)


def test_conservation_scores_bounded():
    out = compute_conservation_scores(["ACGT", "ACGA", "ACGT"])

    assert all(0 <= c <= 1 for _, c in out)
    assert plot_sequence_logo(["ACGT", "ACGT"]) is not None


def test_the_bar_draws_the_score_each_column_reports():
    sequences = ["ACGT"] * 8

    fig = plot_conservation_bar(sequences)

    ax = fig.axes[0]
    heights = [patch.get_height() for patch in ax.patches]
    expected = [value for _pos, value in compute_conservation_scores(sequences)]
    assert heights == pytest.approx(expected)
    assert ax.get_ylabel() == "Conservation"


def test_the_combined_plot_puts_the_bar_under_one_shared_x_axis():
    fig = plot_logo_with_conservation(["ACGT", "ACGT"], title="Conserved")

    logo, bar = fig.axes
    assert logo.get_title() == "Conserved"
    assert bar.get_title() == ""
    assert bar.get_ylabel() == "Conservation"
    assert logo.get_shared_x_axes().joined(logo, bar)
    assert logo.get_xlim() == bar.get_xlim()

