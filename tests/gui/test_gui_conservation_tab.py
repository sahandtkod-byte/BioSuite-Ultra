"""The Sequence Logo & Conservation tab: per-position scores, the logo, motif runs.

``_run_cons`` parses one sequence per line, asks
``biosuite.plotting.conservation_plots`` for the scores and the figure, and draws a
``#`` bar per position.  The expected box is rebuilt in the test from the same
``compute_conservation_scores`` call on the same cleaned sequences, so both the
line-by-line numbers and the tab's own upper-casing are checked.
"""
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import conservation_plots

SEQUENCES = ["ACGTACGTACGT", "ACGAACGTACGT", "ACGTACGTACGA", "ACGTACGTACGT"]


@pytest.fixture
def aligned(ui):
    """Four equal-length aligned sequences, with the box and history emptied."""
    ui.set("cons_text", "\n".join(SEQUENCES))
    ui.clear("cons_stats")
    return ui


def _expected_box(scores):
    return "".join(f"Pos {pos:2d}: {score:.3f} {'#' * int(score * 30)}\n"
                   for pos, score in scores).rstrip("\n")


def _shown(ui, name):
    """What a box displays, without the newline every Tk textbox appends."""
    return ui.text(name).rstrip("\n")


def test_the_box_lists_the_conservation_of_every_position(aligned):
    expected = conservation_plots.compute_conservation_scores(SEQUENCES)
    assert len(expected) == len(SEQUENCES[0]), "one score per alignment column"

    aligned.app._run_cons()

    assert _shown(aligned, "cons_stats") == _expected_box(expected)
    assert aligned.messages == []


def test_the_tab_cleans_the_sequences_before_scoring(aligned):
    """Lower case and stray whitespace are the user's, the scores must still match."""
    padded = "\n".join(f"  {s.lower()}  " for s in SEQUENCES)
    aligned.set("cons_text", padded)
    expected = conservation_plots.compute_conservation_scores(SEQUENCES)

    aligned.app._run_cons()

    assert _shown(aligned, "cons_stats") == _expected_box(expected)


def test_the_logo_figure_is_displayed(aligned):
    plt.close("all")

    aligned.app._run_cons()

    assert aligned.plots, "the logo was never handed to the viewer"
    title, fig = aligned.plots[-1]
    assert title == "Conservation Analysis"
    assert fig.axes, "the conservation figure is empty"


def test_an_empty_alignment_is_refused(ui):
    ui.clear("cons_text")

    ui.app._run_cons()

    assert ui.messages_of("warning") == ["Enter aligned sequences first."]
    assert ui.plots == []


def test_a_scoring_failure_is_reported_without_a_partial_box(aligned, monkeypatch):
    def broken(sequences, alphabet='ACGT'):
        raise RuntimeError("sequences must be the same length")

    monkeypatch.setattr(conservation_plots, "compute_conservation_scores", broken)

    aligned.app._run_cons()

    assert aligned.messages_of("error") == ["sequences must be the same length"]
    assert aligned.text("cons_stats").strip() == ""


def test_the_motifs_typed_in_the_dialog_reach_the_plotter(aligned, monkeypatch):
    real = conservation_plots.plot_motif_enrichment
    seen = {}

    def spy(sequences, motifs, *args, **kwargs):
        seen['sequences'] = sequences
        seen['motifs'] = motifs
        return real(sequences, motifs, *args, **kwargs)

    monkeypatch.setattr(conservation_plots, "plot_motif_enrichment", spy)
    aligned.input_answer = "ATG, CG ,GCG"
    plt.close("all")

    aligned.app._run_motif()

    assert seen['motifs'] == ["ATG", "CG", "GCG"], "the list was not split and stripped"
    assert seen['sequences'] == SEQUENCES


def test_a_motif_run_displays_its_figure(aligned):
    aligned.input_answer = "ATG,CG"
    plt.close("all")

    aligned.app._run_motif()

    assert aligned.plots, "no motif figure was displayed"
    title, fig = aligned.plots[-1]
    assert title == "Motif Enrichment"
    assert fig.axes, "the motif figure has no axes"


def test_a_cancelled_motif_dialog_draws_nothing(aligned):
    aligned.input_answer = ""
    plt.close("all")

    aligned.app._run_motif()

    assert aligned.plots == []
    assert aligned.messages == []


def test_motif_enrichment_without_sequences_asks_for_them(ui):
    ui.clear("cons_text")

    ui.app._run_motif()

    assert ui.messages_of("warning") == ["Enter aligned sequences first."]
    assert ui.asked == [], "no motif dialog may open over an empty box"


def test_a_motif_failure_is_reported_with_its_reason(aligned, monkeypatch):
    def broken(sequences, motifs, *args, **kwargs):
        raise RuntimeError("no motifs to score")

    aligned.input_answer = "ATG"
    monkeypatch.setattr(conservation_plots, "plot_motif_enrichment", broken)

    aligned.app._run_motif()

    assert aligned.messages_of("error") == ["no motifs to score"]


def test_the_sidebar_entries_land_on_the_conservation_frame(aligned):
    aligned.app._gui_seq_logo()
    assert aligned.app._current_frame == "conservation"
    assert aligned.text("cons_stats").strip()

    aligned.clear("cons_stats")
    aligned.app._gui_conservation_bar()
    assert aligned.app._current_frame == "conservation"
    assert aligned.text("cons_stats").strip()
