"""What the `utils.py` report and annotation helpers do when the data cannot answer.

These functions sit under the plots, so their failure mode is the user's problem:
a missing column raised `KeyError`, a single group raised
`TypeError: At least two samples are required; got 1.`, a constant series drew
`t-test p = nan` onto the figure as a result, and one-component PCA raised
`IndexError: list index out of range`.  Each now says what is missing in the log and
leaves the figure alone.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from biosuite.core import utils


@pytest.fixture
def axes():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        yield ax
    finally:
        plt.close(fig)


def texts(ax):
    return [t.get_text() for t in ax.texts]


# ── scatter ─────────────────────────────────────────────────────────────────

def test_two_series_of_different_lengths_are_not_correlated(caplog):
    with caplog.at_level(logging.INFO):
        assert utils.report_scatter_stats([1, 2, 3], [1, 2]) is None
    assert [r.getMessage() for r in caplog.records if r.name == 'biosuite.core.utils'] == [
        "Correlation needs equal-length series, got 3 and 2 values."]


def test_a_constant_series_reports_nothing_instead_of_nan(caplog):
    """scipy answers NaN (plus its own warning) for a flat series; the old code
    printed "Pearson r = nan, p-value = nan" as the analysis result."""
    with caplog.at_level(logging.INFO):
        assert utils.report_scatter_stats([1, 1], [1, 2]) is None
    assert "Pearson correlation is not defined" in caplog.text


def test_a_real_correlation_is_still_reported(caplog):
    with caplog.at_level(logging.INFO):
        stats = utils.report_scatter_stats([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert stats['r'] == pytest.approx(1.0)
    assert "Pearson r = 1.000" in caplog.text


# ── boxplot statistics ──────────────────────────────────────────────────────

def test_a_frame_without_the_named_columns_is_reported_not_keyerror(caplog):
    frame = pd.DataFrame({'other': [1.0, 2.0]})
    with caplog.at_level(logging.INFO):
        assert utils.report_boxplot_stats(frame, 'grp', 'val') == {}
    assert "Boxplot statistics needs the columns ['grp', 'val']" in caplog.text


def test_a_frame_whose_groups_have_no_values_logs_that_rather_than_printing_nothing(caplog):
    frame = pd.DataFrame({'grp': ['a', 'b'], 'val': [np.nan, np.nan]})
    with caplog.at_level(logging.INFO):
        assert utils.report_boxplot_stats(frame, 'grp', 'val') == {}
    assert "no group had a value to summarise" in caplog.text


# ── PCA ─────────────────────────────────────────────────────────────────────

class _OneComponentPCA:
    explained_variance_ratio_ = [0.9]


class _NoComponentsPCA:
    explained_variance_ratio_ = []


def test_a_single_component_pca_does_not_reach_for_a_second_one(caplog):
    with caplog.at_level(logging.INFO):
        assert utils.report_pca_stats(_OneComponentPCA()) == {'var1': 0.9}
    assert "PC1 explains 90.00% of variance" in caplog.text
    assert "PC2" not in caplog.text


def test_a_pca_that_reports_no_variance_ratios_says_so(caplog):
    with caplog.at_level(logging.INFO):
        assert utils.report_pca_stats(_NoComponentsPCA()) == {}
    assert "reports no variance ratios" in caplog.text


def test_a_two_component_pca_reports_both_ratios(caplog):
    class _Two:
        explained_variance_ratio_ = [0.6, 0.3, 0.1]

    with caplog.at_level(logging.INFO):
        assert utils.report_pca_stats(_Two()) == {'var1': 0.6, 'var2': 0.3}
    assert "PC2 explains 30.00%" in caplog.text


# ── significance annotations ────────────────────────────────────────────────

def test_one_group_gets_no_test_and_no_crash(caplog, axes):
    frame = pd.DataFrame({'g': ['a', 'a'], 'v': [1.0, 2.0]})
    with caplog.at_level(logging.INFO):
        utils.add_ttest_to_boxplot(frame, 'g', 'v', axes)
    assert texts(axes) == []
    assert "No significance test for 1 group" in caplog.text


def test_a_group_of_nothing_but_missing_values_is_not_announced_as_nan(caplog, axes):
    frame = pd.DataFrame({'g': ['a', 'b'], 'v': [1.0, np.nan]})
    with caplog.at_level(logging.INFO):
        utils.add_ttest_to_boxplot(frame, 'g', 'v', axes)
    assert texts(axes) == []
    assert "fewer than 2 usable values" in caplog.text


def test_two_groups_still_get_their_ttest(axes):
    frame = pd.DataFrame({'g': ['a', 'a', 'b', 'b'], 'v': [1.0, 2.0, 5.0, 6.0]})
    utils.add_ttest_to_boxplot(frame, 'g', 'v', axes)
    assert len(texts(axes)) == 1
    assert texts(axes)[0].startswith("t-test p = ")


def test_three_groups_still_get_their_anova(axes):
    frame = pd.DataFrame({'g': ['a', 'a', 'b', 'b', 'c', 'c'],
                          'v': [1.0, 2.0, 5.0, 6.0, 9.0, 8.0]})
    utils.add_ttest_to_boxplot(frame, 'g', 'v', axes)
    assert texts(axes)[0].startswith("ANOVA p = ")


# ── regression annotation ───────────────────────────────────────────────────

def test_identical_x_values_draw_no_line_equation(caplog, axes):
    with caplog.at_level(logging.INFO):
        utils.add_regression_eq([1, 1], [3, 4], axes)
    assert texts(axes) == []
    assert "at least two distinct x values" in caplog.text


def test_one_point_draws_no_line_equation(caplog, axes):
    with caplog.at_level(logging.INFO):
        utils.add_regression_eq([1], [3], axes)
    assert texts(axes) == []
    assert "Insufficient data for a regression line." in caplog.text


def test_unequal_series_are_named_instead_of_reaching_scipy(caplog, axes):
    with caplog.at_level(logging.INFO):
        utils.add_regression_eq([1, 2, 3], [1, 2], axes)
    assert texts(axes) == []
    assert "got 3 and 2 values." in caplog.text


def test_missing_pairs_are_dropped_before_the_fit(axes):
    utils.add_regression_eq([1.0, 2.0, 3.0, 4.0], [2.0, np.nan, 6.0, 8.0], axes)
    assert texts(axes)[0].startswith("y = ")
    assert "nan" not in texts(axes)[0]


def test_a_fitted_line_is_written_with_r_squared_and_p(axes):
    utils.add_regression_eq([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], axes)
    assert texts(axes)[0] == "y = 2.000x + 0.000\nR² = 1.000, p = 0.0000"


def test_a_flat_y_series_is_not_annotated_as_a_fit(caplog, axes):
    """Two usable points with the same value: scipy answers with a NaN coefficient
    of determination rather than raising, so the finite check is what keeps the
    figure honest."""
    with caplog.at_level(logging.INFO):
        utils.add_regression_eq([1.0, 2.0], [3.0, 3.0], axes)
    assert texts(axes) == []
    assert "Regression did not produce finite coefficients" in caplog.text


def test_a_zero_variance_pair_is_not_annotated_as_a_test(caplog, axes):
    """Both groups have usable values, so the size guard passes; the statistic
    itself is undefined, and that is the case the finite check exists for."""
    frame = pd.DataFrame({'g': ['a', 'a', 'b', 'b'], 'v': [1.0, 1.0, 1.0, 1.0]})
    with caplog.at_level(logging.INFO):
        utils.add_ttest_to_boxplot(frame, 'g', 'v', axes)
    assert texts(axes) == []
    assert "did not produce a finite p-value" in caplog.text
