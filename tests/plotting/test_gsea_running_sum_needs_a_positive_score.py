"""A GSEA running sum divides by the positive scores, so there must be one.

``np.sum(scores[scores > 0]) + 1e-10`` is undefined when no score is positive: the
probe measured a curve between -4.0e12 and -1.3e9 whose own title read
``max ES = -1257302210.934``.  A ranked list with nothing ranked above zero has
no enrichment to plot, and the demo says so instead of drawing it.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import specialized_plots as sp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_ranking_with_no_positive_score_is_refused():
    scores = -np.abs(np.random.default_rng(0).normal(size=50))

    with pytest.raises(ValueError) as excinfo:
        sp._running_sum(scores)

    message = str(excinfo.value)
    assert "no positive score" in message, message
    assert "nothing to enrich" in message, message


def test_the_running_sum_is_the_cumulative_score_over_the_positive_total():
    """The arithmetic, pinned: cumsum([3, 2, -1]) / 5."""
    assert np.round(sp._running_sum(np.array([3.0, 2.0, -1.0])), 6).tolist() == \
        [0.6, 1.0, 0.8]


def test_a_single_positive_score_is_enough():
    assert sp._running_sum(np.array([0.0, 2.0, -5.0])).tolist() == [0.0, 1.0, -1.5]


def test_the_demo_says_so_and_saves_nothing_for_an_all_negative_file(run_demo, tmp_path):
    ranked = tmp_path / "down.csv"
    pd.DataFrame({"score": [-0.5, -1.0, -2.0]}).to_csv(ranked, index=False)

    report = run_demo(sp.gsea_plot, answers={
        "load ranked list from file": "y",
        "file path": str(ranked),
        "score column": "score",
    })

    assert "Error:" in report["stdout"], report["stdout"]
    assert "no positive score" in report["stdout"], report["stdout"]
    assert report["images"] == [], "a refused curve must not be saved"
    assert report["figures_open"] == 0


def test_a_positive_ranking_still_draws_and_saves(run_demo, tmp_path):
    """The positive control for the refusal above."""
    ranked = tmp_path / "up.csv"
    pd.DataFrame({"score": [3.0, 2.0, -1.0]}).to_csv(ranked, index=False)

    report = run_demo(sp.gsea_plot, answers={
        "load ranked list from file": "y",
        "file path": str(ranked),
        "score column": "score",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["gsea.png"]
    # cumsum([3, 2, -1]) / 5 = [0.6, 1.0, 0.8]: the peak is the second point.
    assert report["figure_title"] == "GSEA Running Sum (max ES = 1.000)", \
        report["figure_title"]
