"""Math demos: the synthetic curves save a PNG, and typed parameters reach the plot.

A demo that fails after drawing its figure must release it, and a curve that
overflows to infinity must be refused before a figure exists at all.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import math_plots as mp

CURVE_DEMOS = [
    ("sine_plot", "sine.png"),
    ("cosine_plot", "cosine.png"),
    ("linear_plot", "linear.png"),
    ("quadratic_plot", "quadratic.png"),
    ("cubic_plot", "cubic.png"),
    ("exponential_plot", "exponential.png"),
    ("logistic_plot", "logistic.png"),
]


@pytest.mark.parametrize("name,filename", CURVE_DEMOS)
def test_default_parameters_save_a_plot(run_demo, name, filename):
    report = run_demo(getattr(mp, name))
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == [filename]
    assert report["empty_images"] == []
    assert report["figures_open"] == 0


@pytest.fixture
def captured_titles(monkeypatch):
    """Record the title of the figure each demo hands to ``ask_save_plot``.

    The demos close their figure as part of saving, so the title has to be read
    while it is still current — it is also the only place the typed parameters
    surface, since the numbers go into the plotted expression.
    """
    titles = []
    real = mp.ask_save_plot

    def spy(default_name, save_format, dpi, pdf=None, story=None):
        titles.append(plt.gca().get_title())
        return real(default_name, save_format, dpi, pdf, story)

    monkeypatch.setattr(mp, "ask_save_plot", spy)
    return titles


def test_typed_parameters_reach_the_curve(run_demo, captured_titles):
    report = run_demo(mp.sine_plot, quiet=False, answers={
        "amplitude a": "2",
        "frequency b": "3",
        "phase c": "0.5",
    })
    assert captured_titles == ["Sine: y = 2.0·sin(3.0x + 0.5)"]
    assert report["images"] == ["sine.png"]
    assert report["figures_open"] == 0


def test_unparseable_parameter_falls_back_to_the_default(run_demo, captured_titles):
    report = run_demo(mp.sine_plot, quiet=False, answers={
        "amplitude a": "wide",
        "frequency b": "",
        "phase c": "",
    })
    assert captured_titles == ["Sine: y = 1·sin(1x + 0)"]
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["sine.png"]


def test_a_failure_after_the_figure_is_drawn_closes_it(run_demo, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mp, "ask_save_plot", broken)

    report = run_demo(mp.sine_plot)

    assert "Error: disk full" in report["stdout"]
    assert report["figures_open"] == 0, "the failed demo left its figure pinned"
    assert report["images"] == []


@pytest.mark.parametrize("name,answers", [
    ("exponential_plot", {"scale a": "1", "rate b": "1000", "offset c": "0"}),
    ("quadratic_plot", {"a (default": "1e308", "b (default": "0", "c (default": "0"}),
    ("cubic_plot", {"a (default": "1e308", "b (default": "0",
                    "c (default": "0", "d (default": "0"}),
])
def test_a_curve_that_goes_infinite_is_refused_before_a_figure(run_demo, name, answers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = run_demo(getattr(mp, name), quiet=False, answers=answers)

    assert "Error:" in report["stdout"]
    assert "not finite" in report["stdout"], report["stdout"]
    assert report["figures_open"] == 0
    assert report["images"] == []


def test_a_steep_logistic_is_computed_without_overflow(run_demo, monkeypatch):
    captured = {}

    def capture(*args, **kwargs):
        fig = plt.gcf()
        captured["y"] = fig.axes[0].lines[0].get_ydata().copy()
        plt.close(fig)

    monkeypatch.setattr(mp, "ask_save_plot", capture)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_demo(mp.logistic_plot, quiet=False, answers={
            "carrying capacity l": "1",
            "growth rate k": "1000",
            "midpoint x0": "0",
        })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert "y" in captured, "the logistic figure was never drawn"
    assert np.all(np.isfinite(captured["y"]))
    assert captured["y"].min() >= 0.0 and captured["y"].max() <= 1.0
    overflow = [w for w in caught if "overflow" in str(w.message)]
    assert overflow == [], f"the stable logistic still overflowed: {overflow}"
