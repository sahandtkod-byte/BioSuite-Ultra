"""The Synteny tab: gene orders in, collinearity score and dotplot out.

``_run_synteny`` splits the two entries, asks ``biosuite.plotting.synteny`` for the
score and the pairs, and writes the counts plus a dotplot.  The expected text is
rebuilt in the test from the same ``compute_synteny_score`` call on the same two
lists, so the numbers are the module's and the splitting is the tab's.
"""
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import synteny

GENOME_1 = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
GENOME_2 = ["GeneA", "GeneC", "GeneB", "GeneE", "GeneD"]


@pytest.fixture
def gene_orders(ui):
    ui.set("syn_g1", ",".join(GENOME_1))
    ui.set("syn_g2", ",".join(GENOME_2))
    ui.clear("syn_stats")
    return ui


def test_the_box_reports_the_score_and_the_orders_the_core_used(gene_orders):
    score, pairs = synteny.compute_synteny_score(GENOME_1, GENOME_2)
    assert 0 <= score <= 1 and pairs, "the oracle itself must produce a score"

    gene_orders.app._run_synteny()

    text = gene_orders.text("syn_stats")
    assert f"Synteny score: {score:.3f}\n" in text, text
    assert f"Genome 1: {len(GENOME_1)} genes\n" in text, text
    assert f"Genome 2: {len(GENOME_2)} genes\n" in text, text
    assert f"Common: {len(set(GENOME_1) & set(GENOME_2))}\n" in text, text
    assert gene_orders.messages == []


def test_the_entries_are_split_stripped_and_emptied(gene_orders):
    score, _pairs = synteny.compute_synteny_score(GENOME_1, GENOME_2)
    gene_orders.set("syn_g1", " GeneA , GeneB ,, GeneC , GeneD , GeneE ")

    gene_orders.app._run_synteny()

    assert f"Synteny score: {score:.3f}\n" in gene_orders.text("syn_stats")
    assert gene_orders.messages == []


def test_the_dotplot_is_displayed(gene_orders):
    plt.close("all")

    gene_orders.app._run_synteny()

    assert gene_orders.plots, "the dotplot was never handed to the viewer"
    title, fig = gene_orders.plots[-1]
    assert title == "Synteny Dotplot"
    assert fig.axes, "the dotplot figure is empty"


def test_a_genome_without_genes_is_refused(gene_orders):
    gene_orders.set("syn_g1", "")

    gene_orders.app._run_synteny()

    assert gene_orders.messages_of("warning") == ["Enter gene orders for both genomes."]
    assert gene_orders.plots == []


def test_the_second_genome_matters_too(gene_orders):
    gene_orders.set("syn_g2", "   ")

    gene_orders.app._run_synteny()

    assert gene_orders.messages_of("warning") == ["Enter gene orders for both genomes."]


def test_a_scoring_failure_is_named_in_a_dialog(gene_orders, monkeypatch):
    def broken(genes1, genes2):
        raise RuntimeError("no genes in common")

    monkeypatch.setattr(synteny, "compute_synteny_score", broken)

    gene_orders.app._run_synteny()

    assert gene_orders.messages_of("error") == ["no genes in common"]
    assert gene_orders.text("syn_stats").strip() == ""


def test_the_sidebar_entry_shows_the_frame_and_runs_the_analysis(gene_orders):
    plt.close("all")

    gene_orders.app._gui_synteny()

    assert gene_orders.app._current_frame == "syntenytabs"
    assert gene_orders.text("syn_stats").strip()
    assert gene_orders.plots
