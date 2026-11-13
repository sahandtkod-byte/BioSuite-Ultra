"""Synteny edges: too-short words, marginal histograms, a passed-in axes, inversions.

``plot_dotplot`` can build its own figure with two marginal histograms, reuse an
axes the caller owns, or skip the histograms entirely; each of those shapes is a
different set of axes on the returned figure.  ``plot_synteny_dotplot`` colours a
gene pair red only when it found an inversion, and ``compute_synteny_score`` has to
refuse two genomes that share fewer than two genes.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.synteny import (
    compute_dotplot,
    compute_synteny_score,
    plot_dotplot,
    plot_synteny,
    plot_synteny_dotplot,
)

SHARED = "ACGTACGTGGCCTTAA" * 3

#: The two colours ``plot_synteny_dotplot`` picks, as matplotlib resolves them.
COLLINEAR = (0.0, 1.0, 0.533)
INVERTED = (1.0, 0.42, 0.42)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rgb(collection):
    """The face colour of every marker, without the alpha channel."""
    return [tuple(round(float(value), 3) for value in rgba[:3])
            for rgba in collection.get_facecolor()]


def test_a_word_longer_than_either_sequence_finds_nothing():
    assert compute_dotplot("ACGT", "ACGT", word_size=10) == []
    assert compute_dotplot(SHARED, "ACGT", word_size=5) == []


def test_a_word_exactly_as_long_as_a_sequence_still_matches_it():
    """One k-mer is a whole read, so the shorter sequence contributes a single column."""
    assert compute_dotplot(SHARED, "ACGT", word_size=4) == [
        (0, 0), (4, 0), (16, 0), (20, 0), (32, 0), (36, 0)]


def test_dotplot_with_histograms_builds_three_axes():
    fig = plot_dotplot(SHARED, SHARED, word_size=6, show_histograms=True)

    assert len(fig.axes) == 3, "the dot panel plus one histogram on each side"
    dots, hist_x, hist_y = fig.axes
    assert len(dots.collections) == 1, "the matches are drawn as one scatter"
    assert len(hist_x.patches) > 0, "the top histogram counts the matched x positions"
    assert len(hist_y.patches) > 0, "the side histogram counts the matched y positions"
    assert hist_x.get_ylabel() == "Count"
    assert hist_y.get_xlabel() == "Count"


def test_dotplot_without_histograms_draws_a_single_panel():
    fig = plot_dotplot(SHARED, SHARED, word_size=6, show_histograms=False)

    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == 0, "no histogram was asked for"


def test_dotplot_on_a_given_axes_does_not_invent_histogram_axes():
    outer, ax = plt.subplots(figsize=(4, 4))

    fig = plot_dotplot(SHARED, SHARED, word_size=6, ax=ax, show_histograms=True)

    assert fig is outer, "the caller's figure is returned, not a new one"
    assert len(outer.axes) == 1, "an external axes has no histogram neighbours to fill"
    assert len(ax.collections) == 1


def test_dotplot_of_sequences_with_no_shared_word_draws_an_empty_panel():
    fig = plot_dotplot("AAAAAAAA", "CCCCCCCC", word_size=4, show_histograms=True)

    assert len(fig.axes) == 3, "the histogram axes are still allocated"
    assert len(fig.axes[0].collections) == 0, "no match means no scatter, and nothing is invented"
    assert fig.axes[0].get_xlim() == (0.0, 8.0), "the panel still spans both sequences"
    assert fig.axes[0].get_xlabel() == "Sequence 1"


def test_two_genomes_sharing_one_gene_have_no_collinearity_to_score():
    score, pairs = compute_synteny_score(["A", "B", "C"], ["A", "X", "Y"])

    assert (score, pairs) == (0.0, []), "one shared gene cannot form a pair"


def test_identical_gene_orders_score_one():
    genes = ["A", "B", "C", "D"]

    score, pairs = compute_synteny_score(genes, genes)

    assert score == pytest.approx(1.0)
    assert len(pairs) == len(genes) * (len(genes) - 1) // 2, "every ordered pair agrees"


def test_an_inverted_pair_is_marked_in_red():
    forward = ["GeneA", "GeneB", "GeneC", "GeneD"]
    flipped = ["GeneA", "GeneD", "GeneC", "GeneB"]

    fig = plot_synteny_dotplot(forward, flipped, title="Inverted")

    points = fig.axes[0].collections[0]
    assert len(points.get_offsets()) == 4, "one dot per shared gene"
    assert _rgb(points).count(COLLINEAR) == 1, "only the leading gene keeps its order"
    assert _rgb(points).count(INVERTED) == 3, "every gene the inversion moved turns red"


def test_a_collinear_pair_is_never_marked_in_red():
    genes = ["GeneA", "GeneB", "GeneC", "GeneD"]

    fig = plot_synteny_dotplot(genes, genes)

    points = fig.axes[0].collections[0]
    assert set(_rgb(points)) == {COLLINEAR}, "an unshuffled genome has no non-collinear gene"


def test_a_passed_in_axes_keeps_the_dotplot_on_the_callers_figure():
    outer, ax = plt.subplots()

    fig = plot_synteny_dotplot(["A", "B"], ["A", "B"], ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1


def test_plot_synteny_shows_both_gene_orders_and_the_links():
    fig = plot_synteny(["A", "B", "C"], ["C", "B", "A"],
                       genome1_name="Human", genome2_name="Mouse")

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Human", "Mouse"]
    assert len(fig.axes[0].patches) == 3, "one block per gene in the first order"
    names = [t.get_text() for t in fig.axes[0].texts if t.get_text()]
    assert names == ["A", "B", "C"]
    links = [t for t in fig.axes[0].texts if not t.get_text()]
    assert len(links) == 3, "one line for each of the three shared genes"


def test_a_reversed_order_scores_zero():
    order = ["A", "B", "C", "D"]

    score, pairs = compute_synteny_score(order, list(reversed(order)))

    assert score == pytest.approx(0.0)
    assert pairs == [], "not one pair keeps its order"


def test_a_no_orthologs_panel_still_names_both_axes():
    fig = plot_synteny_dotplot(["A", "B"], ["X", "Y"],
                               genome1_name="Genome A", genome2_name="Genome B")

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Genome A"
    assert ax.get_ylabel() == "Genome B"
    assert [t.get_text() for t in ax.texts] == ["No orthologs found"]


def test_more_than_twenty_shared_genes_label_a_stated_subset():
    genes = [f"G{i:02d}" for i in range(30)]

    fig = plot_synteny_dotplot(genes, genes, title="Shared")

    labelled = [t.get_text() for t in fig.axes[0].texts if t.get_text()]
    assert len(labelled) == 20
    assert fig.axes[0].get_title() == "Shared (labels for 20 of 30 genes)"


def test_a_map_with_no_shared_genes_draws_both_orders_without_links():
    fig = plot_synteny(["A", "B"], ["C", "D"])

    assert len(fig.axes) == 2
    assert [len(ax.patches) for ax in fig.axes] == [2, 2]
    assert [t for t in fig.axes[0].texts if not t.get_text()] == [], \
        "there is no shared gene to draw a line for"
    grey = mcolors.to_rgba('lightgray')
    for ax in fig.axes:
        assert all(tuple(p.get_facecolor()) == grey for p in ax.patches)

