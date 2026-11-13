"""One call cannot mean two different things depending on which backend draws it.

``interactive_manhattan`` drew its significance threshold in plotly and dropped it
in the matplotlib fallback (measured: zero reference lines), while
``interactive_ma`` drew the zero line in the fallback and not in plotly.  A user
who saved a PNG and a user who looked at the GUI were reading two different
claims about the same data.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip

CHROMS = ["1", "1", "2", "2"]
POSITIONS = [10, 20, 10, 20]
NEG_LOG_P = [2.0, 5.0, 3.0, 1.0]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def fallback():
    """Force the matplotlib branch for the duration of one test."""
    saved = ip.HAS_PLOTLY
    ip.HAS_PLOTLY = False
    yield
    ip.HAS_PLOTLY = saved


def test_manhattan_draws_the_threshold_in_plotly():
    fig = ip.interactive_manhattan(CHROMS, POSITIONS, NEG_LOG_P, threshold=0.01)

    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].y0 == pytest.approx(2.0), "-log10(0.01) is 2"


def test_manhattan_draws_the_same_threshold_in_the_fallback(fallback):
    fig = ip.interactive_manhattan(CHROMS, POSITIONS, NEG_LOG_P, threshold=0.01)

    lines = fig.axes[0].lines
    assert len(lines) == 1, "the fallback lost the threshold line"
    assert lines[0].get_ydata()[0] == pytest.approx(2.0)
    assert lines[0].get_label() == "p=0.01"


def test_ma_draws_the_zero_line_in_plotly():
    fig = ip.interactive_ma([1.0, 2.0], [0.5, 2.0])

    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].y0 == 0


def test_ma_draws_the_same_zero_line_in_the_fallback(fallback):
    fig = ip.interactive_ma(np.array([1.0, 2.0]), np.array([0.5, 2.0]))

    assert [line.get_ydata()[0] for line in fig.axes[0].lines] == [0]


@pytest.mark.parametrize("threshold", [0.0, 1.5, -0.1])
def test_a_threshold_that_is_not_a_p_value_is_refused_by_name(threshold):
    """0.0 used to vanish (``if threshold:``), 1.5 drew a negative line and -0.1 drew NaN."""
    for backend_has_plotly in (True, False):
        ip.HAS_PLOTLY = backend_has_plotly
        try:
            with pytest.raises(ValueError, match="threshold is a p-value and must be "
                                                "between 0 and 1 exclusive"):
                ip.interactive_manhattan(CHROMS, POSITIONS, NEG_LOG_P,
                                         threshold=threshold)
        finally:
            ip.HAS_PLOTLY = True


def test_no_threshold_means_no_line_in_either_backend():
    """The positive control for the three refusals: None is a legal silence."""
    plotly_fig = ip.interactive_manhattan(CHROMS, POSITIONS, NEG_LOG_P)
    assert len(plotly_fig.layout.shapes) == 0
    plt.close("all")
    saved = ip.HAS_PLOTLY
    ip.HAS_PLOTLY = False
    try:
        plain = ip.interactive_manhattan(CHROMS, POSITIONS, NEG_LOG_P)
    finally:
        ip.HAS_PLOTLY = saved
    assert len(plain.axes[0].lines) == 0
