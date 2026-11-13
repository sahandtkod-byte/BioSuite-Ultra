"""A Q-Q plot of p-values is a Q-Q plot of probabilities, and nothing else fits.

``interactive_qq`` used to clip whatever it was given into [1e-16, 1-1e-16], so a
p-value of 1.4 was drawn at the far end of the diagonal as if it were the most
extreme possible result, and the same call through ``plot_api.qqplot`` refused it
with a sentence.  Two Q-Q plots in one package cannot disagree about what a
p-value is; the check here is the one ``plot_api`` already applies.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip
from biosuite.plotting import plot_api


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("bad", [1.4, -0.2, np.nan, np.inf, -np.inf])
def test_a_value_that_is_not_a_probability_is_refused(bad):
    with pytest.raises(ValueError) as excinfo:
        ip.interactive_qq([0.5, bad])

    message = str(excinfo.value)
    assert "1 of 2 p-values outside [0, 1]" in message, message


def test_the_refusal_says_why_clipping_would_be_wrong():
    with pytest.raises(ValueError) as excinfo:
        ip.interactive_qq([1.4])

    assert "clipping it would draw the impossible value" in str(excinfo.value)


def test_the_same_input_is_refused_by_the_plot_api_qq_plot():
    """The two Q-Q plots are held to one contract, in both directions."""
    for bad in (1.4, -0.2, np.nan):
        with pytest.raises(ValueError):
            plot_api.qqplot([0.5, bad])
        with pytest.raises(ValueError):
            ip.interactive_qq([0.5, bad])


@pytest.mark.parametrize("p", [[0.0], [1.0], [0.001, 0.5, 0.999]])
def test_the_edges_of_the_range_are_probabilities(p):
    """The positive control: 0 and 1 are legal and are clipped only to stay finite."""
    fig = ip.interactive_qq(p)
    assert len(fig.data) == 2  # the points plus the diagonal
    assert len(fig.data[0].x) == len(p)


def test_the_diagonal_is_drawn_through_the_same_extremes():
    fig = ip.interactive_qq([0.01, 0.3, 0.6, 0.95])
    reference = fig.data[1]

    assert list(reference.x) == list(reference.y), "the reference line is y = x"


def test_the_fallback_draws_the_same_points_and_line():
    ip.HAS_PLOTLY = False
    try:
        fig = ip.interactive_qq(np.array([0.01, 0.3, 0.6, 0.95]))
    finally:
        ip.HAS_PLOTLY = True

    assert len(fig.axes[0].collections[0].get_offsets()) == 4
    assert len(fig.axes[0].lines) == 1
