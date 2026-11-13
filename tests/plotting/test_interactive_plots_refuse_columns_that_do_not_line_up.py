"""Every column of a plot has to describe the same entries, and the message says which.

These functions used to absorb a mismatch differently per backend and per
function: a short ``labels`` list raised ``IndexError`` in plotly and was ignored
by the fallback, a short ``row_labels`` list silently mislabelled the rows, and a
three-long ``errors`` list against two bars died in matplotlib's ``errorbar`` with
numpy's own accounting.  One refusal, with the lengths in it, replaces all three.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _refusal(*args, **kwargs):
    with pytest.raises(ValueError) as excinfo:
        ip.interactive_scatter(*args, **kwargs)
    return str(excinfo.value)


def test_a_label_list_has_one_entry_per_point():
    text = _refusal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["a"])

    assert "x=3, labels=1" in text, text


def test_a_colour_column_has_one_entry_per_point():
    text = _refusal([1.0, 2.0], [1.0, 2.0], color_col=["a", "b", "c"])

    assert "x=2, color_col=3" in text, text


def test_the_headers_are_checked_by_the_same_error_bar_lengths():
    with pytest.raises(ValueError, match="categories=2, errors=3"):
        ip.interactive_bar(["a", "b"], [1.0, 2.0], errors=[1.0, 2.0, 3.0])


def test_every_line_needs_one_point_per_x():
    with pytest.raises(ValueError, match="x=3, ys\\[1\\]=2"):
        ip.interactive_line([1, 2, 3], [[1, 2, 3], [4, 5]])


def test_a_heatmap_needs_two_dimensions():
    with pytest.raises(ValueError, match="needs a 2-D matrix, got a 1-D array"):
        ip.interactive_heatmap([1.0, 2.0, 3.0])


def test_a_heatmap_row_label_list_is_one_per_row():
    with pytest.raises(ValueError, match="got 1 row labels for 2 rows"):
        ip.interactive_heatmap([[1.0, 2.0], [3.0, 4.0]], row_labels=["r1"])


def test_a_heatmap_column_label_list_is_one_per_column():
    with pytest.raises(ValueError, match="got 1 column labels for 2 columns"):
        ip.interactive_heatmap([[1.0, 2.0], [3.0, 4.0]], col_labels=["c1"])


def test_the_volcano_gene_names_line_up_with_the_points():
    with pytest.raises(ValueError, match="lfc=3, gene_names=2"):
        ip.interactive_volcano([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], gene_names=["a", "b"])


def test_a_negative_fold_change_floor_is_refused():
    with pytest.raises(ValueError, match="fc_thresh must be a fold change of 0 or more"):
        ip.interactive_volcano([1.0], [0.01], fc_thresh=-1)


@pytest.mark.parametrize("p_thresh", [0.0, -0.05, 1.5])
def test_a_p_threshold_outside_the_probability_range_is_refused(p_thresh):
    with pytest.raises(ValueError, match="p_thresh must be a p-value above 0"):
        ip.interactive_volcano([1.0], [0.01], p_thresh=p_thresh)


def test_ma_gene_names_line_up_with_the_points():
    with pytest.raises(ValueError, match="log_fc=3, gene_names=2"):
        ip.interactive_ma([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], gene_names=["a", "b"])


def test_ma_sig_is_one_flag_per_point():
    with pytest.raises(ValueError, match="log_fc=3, sig=2"):
        ip.interactive_ma([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], sig=[True, False])


def test_ma_refuses_a_negative_fold_change_floor():
    with pytest.raises(ValueError, match="fc_thresh must be a fold change of 0 or more"):
        ip.interactive_ma([1.0], [0.1], fc_thresh=-2)


def test_a_pie_of_values_that_cancel_out_is_refused():
    with pytest.raises(ValueError, match="add up to more than zero"):
        ip.interactive_pie(["a", "b"], [5, -5])


@pytest.mark.parametrize("function", [ip.interactive_boxplot, ip.interactive_violin])
def test_a_group_with_no_values_is_named(function):
    with pytest.raises(ValueError, match="group 'empty' has no values"):
        function({"full": [1.0, 2.0], "empty": []})


@pytest.mark.parametrize("function", [ip.interactive_boxplot, ip.interactive_violin])
def test_the_same_call_with_values_still_draws(function):
    """The positive control: one value is enough; none is not."""
    fig = function({"a": [1.0]})
    assert len(fig.data) == 1
