"""Synteny refuses the words, steps and gene orders it cannot draw.

A zero-length word matches every position, a step of zero has no positions at
all, and a gene order that names a gene twice has no single position for it.
Each of those is reported with the value that caused it instead of drawing a
plot that means something different from what was asked.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.synteny import (
    compute_dotplot,
    compute_synteny_score,
    plot_dotplot,
    plot_gene_order,
    plot_synteny,
    plot_synteny_dotplot,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("word_size", [0, -3])
def test_a_word_size_below_one_is_refused(word_size):
    with pytest.raises(ValueError, match="word_size must be at least 1"):
        compute_dotplot("ACGTACGT", "ACGTACGT", word_size=word_size)


def test_a_word_size_that_is_not_a_whole_number_is_named():
    with pytest.raises(ValueError, match="word_size must be a whole number, got '4'"):
        compute_dotplot("ACGTACGT", "ACGTACGT", word_size="4")


def test_a_step_below_one_is_refused():
    with pytest.raises(ValueError, match="step must be at least 1"):
        compute_dotplot("ACGTACGT", "ACGTACGT", word_size=2, step=0)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_hit_budget_below_one_is_refused(limit):
    with pytest.raises(ValueError, match="max_hits must be at least 1"):
        compute_dotplot("ACGTACGT", "ACGTACGT", word_size=2, max_hits=limit)


def test_a_repetitive_pair_stops_at_the_hit_budget():
    with pytest.raises(ValueError, match="more than 3 matches"):
        compute_dotplot("A" * 50, "A" * 50, word_size=5, max_hits=3)


def test_a_sequence_that_is_not_a_string_is_named():
    with pytest.raises(ValueError, match="seq1 must be a string, got list"):
        compute_dotplot(["A"], "ACGT", word_size=1)


def test_a_dotplot_needs_a_base_on_each_axis():
    with pytest.raises(ValueError, match="non-empty sequence on each axis"):
        plot_dotplot("", "ACGT")


@pytest.mark.parametrize("entry_point", [compute_synteny_score, plot_synteny_dotplot, plot_synteny])
def test_every_entry_point_refuses_a_gene_named_twice(entry_point):
    with pytest.raises(ValueError, match="genes1 names 'A' twice"):
        entry_point(["A", "A", "B"], ["A", "B"])


def test_a_gene_that_is_not_a_name_is_reported_by_position():
    with pytest.raises(ValueError, match="genes1 entry 2 is not a gene name: 7"):
        compute_synteny_score(["A", 7], ["A", "B"])


@pytest.mark.parametrize("entry_point", [plot_synteny_dotplot, plot_synteny])
def test_a_plot_needs_a_gene_in_the_first_order(entry_point):
    with pytest.raises(ValueError, match="genes1 is empty"):
        entry_point([], ["A"])


def test_a_synteny_map_needs_a_gene_in_the_second_order():
    with pytest.raises(ValueError, match="genes2 is empty"):
        plot_synteny(["A"], [])


def test_the_score_itself_accepts_two_empty_orders():
    assert compute_synteny_score([], []) == (0.0, [])


def test_a_gene_order_diagram_needs_a_gene():
    with pytest.raises(ValueError, match="at least one gene"):
        plot_gene_order([])


def test_a_gene_order_diagram_names_a_bad_gene():
    with pytest.raises(ValueError, match="genes entry 2 is not a gene name: ''"):
        plot_gene_order(["A", ""])


def test_a_supplied_colour_list_may_not_be_empty():
    with pytest.raises(ValueError, match="at least one colour"):
        plot_gene_order(["A"], colors=[])
