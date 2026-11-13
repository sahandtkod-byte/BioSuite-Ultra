"""A GC window is a number of bases, and the plot is not allowed to divide by it anyway.

``window=0`` produced 157 points starting at x = -1 whose first value was NaN —
the mean of the whole curve went NaN too and the legend read ``Avg nan%`` —
``window=-20`` and ``window=2.5`` raised ``IndexError`` from inside numpy, and a
window longer than the sequence was silently clamped with only the title to
explain that the curve was one point.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import sequence_viewer as sv

#: 144 bases of ordinary mixed DNA.
SEQ = "GCTAGCATGCATGCATGCGATCGATCGTAGCTAGCATCG" * 4


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("window, reason", [
    (0, "window must cover at least one base, got 0"),
    (-20, "window must cover at least one base, got -20"),
    (2.5, "window must be a whole number, got 2.5"),
    (None, "window must be a whole number, got None"),
    ("50", "window must be a whole number, got '50'"),
])
def test_a_window_that_cannot_be_a_span_of_bases_is_refused(window, reason):
    before = set(plt.get_fignums())

    with pytest.raises(ValueError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        sv.draw_gc_content_plot(SEQ, window=window)

    assert set(plt.get_fignums()) == before, "a refused window drew the curve anyway"


def test_an_empty_sequence_is_refused_before_the_mean_of_nothing_is_taken():
    with pytest.raises(ValueError, match="must not be empty"):
        sv.draw_gc_content_plot("", window=10)


def test_no_window_the_tests_ask_for_produces_a_nan_any_more():
    """The curve's own numbers, not the artists around them."""
    for window in (1, 2, 20, 144):
        fig = sv.draw_gc_content_plot(SEQ, window=window)
        y = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        assert len(y) == len(SEQ) - window + 1, window
        assert not np.isnan(y).any(), (window, y[:3])
        assert 0.0 <= y.min() and y.max() <= 100.0, (window, y.min(), y.max())
        plt.close(fig)


def test_a_numpy_integer_window_is_still_an_integer_window():
    fig = sv.draw_gc_content_plot(SEQ, window=np.int64(20))

    assert len(fig.axes[0].lines[0].get_ydata()) == len(SEQ) - 19
    plt.close(fig)


def test_a_window_longer_than_the_sequence_says_both_numbers():
    fig = sv.draw_gc_content_plot("GCATATATAT", window=1000)

    title = fig.axes[0].get_title()
    assert title.startswith("GC Content (window=10"), title
    assert "10 bp" in title and "1,000 bp window asked for" in title, title
    assert len(fig.axes[0].lines[0].get_ydata()) == 1


def test_the_window_the_caller_meant_needs_no_explanation():
    """The positive control for the clamp message above."""
    fig = sv.draw_gc_content_plot("GCATATATAT", window=10)

    assert fig.axes[0].get_title() == "GC Content (window=10)"
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), [20.0])


def test_the_first_base_of_every_window_is_inside_it():
    """A 100-bp window on G + 99 A is 1%, not the 0% an off-by-one used to report."""
    fig = sv.draw_gc_content_plot("G" + "A" * 99, window=100)

    assert np.asarray(fig.axes[0].lines[0].get_ydata())[0] == pytest.approx(1.0)


def test_a_window_of_pure_g_is_a_hundred_percent_across_the_whole_curve():
    fig = sv.draw_gc_content_plot("G" * 100, window=50)

    assert np.allclose(fig.axes[0].lines[0].get_ydata(), 100.0)


def test_the_average_the_mean_line_draws_is_the_average_of_the_curve():
    """The legend quotes the same number the dashed line sits at."""
    fig = sv.draw_gc_content_plot("GCGCGC" + "A" * 94, window=100)

    y = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
    assert y[0] == pytest.approx(6.0), y
    mean_line = fig.axes[0].lines[1]
    mean = float(mean_line.get_ydata()[0])
    assert mean == pytest.approx(float(y.mean()), abs=1e-9), (mean, y.mean())
    assert mean_line.get_label() == f"Avg {mean:.1f}%", mean_line.get_label()
