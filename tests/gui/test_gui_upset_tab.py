"""The UpSet tab: set text in, statistics and a diagram out.

The handler parses ``name:elem1,elem2`` lines, asks
``biosuite.plotting.upset_plots`` for the statistics and the figure, and writes the
numbers into the right-hand box.  The tests build the sets dict first and render it
into the box as text, so the expected statistics are the ones the same core function
returns for the same dict — not a remembered string.
"""
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import upset_plots


@pytest.fixture
def upset_sets(ui):
    """Three sets written the way the box expects them."""
    sets = {
        "Genes_A": {"BRCA1", "TP53", "MYC", "EGFR"},
        "Genes_B": {"TP53", "MYC", "PTEN", "RB1"},
        "Genes_C": {"BRCA1", "PTEN", "APC", "VHL"},
    }
    text = "\n".join(f"{name}:{','.join(sorted(members))}"
                     for name, members in sets.items())
    ui.set("upset_text", text)
    ui.clear("upset_stats")
    return sets


def test_the_stats_box_shows_the_statistics_the_core_computes(ui, upset_sets):
    stats = upset_plots.compute_set_statistics(upset_sets)
    assert stats["total_union"] == 8
    assert stats["total_intersection"] == 0

    ui.app._run_upset()

    text = ui.text("upset_stats")
    assert f"Set sizes: {stats['sizes']}\n" in text, text
    assert f"Union: {stats['total_union']}\n" in text, text
    assert f"Intersection: {stats['total_intersection']}\n" in text, text
    assert f"Unique per set: {stats['unique_per_set']}\n" in text, text
    assert ui.messages == []


def test_the_diagram_is_displayed_after_the_statistics(ui, upset_sets):
    plt.close("all")

    ui.app._run_upset()

    assert ui.plots, "the UpSet figure was never handed to the viewer"
    title, fig = ui.plots[-1]
    assert title == "UpSet Plot"
    assert any("UpSet" in ax.get_title() for ax in fig.axes),             "the diagram has no UpSet axes"


def test_the_upset_plot_does_not_close_the_figures_already_open(ui, upset_sets):
    """The handler ended with a bare ``plt.close()``, which closes *every* figure —
    the one just handed to the viewer included, and every window the history owns."""
    plt.close("all")
    older_fig, _ = plt.subplots()
    ui.app._show_plot_from_figure(older_fig, "Older Plot")

    ui.app._run_upset()

    _title, fig = ui.plots[-1]
    assert plt.fignum_exists(fig.number), "the viewer received a closed figure"
    assert plt.fignum_exists(older_fig.number), "an open window's figure must survive"


def test_an_empty_box_is_refused_before_anything_is_drawn(ui):
    ui.clear("upset_text")

    ui.app._run_upset()

    assert ui.messages_of("warning") == ["Enter set data first."]
    assert ui.plots == []


def test_lines_without_a_colon_are_not_sets(ui):
    ui.set("upset_text", "BRCA1, TP53, MYC\nno colon here at all")

    ui.app._run_upset()

    assert ui.messages_of("warning") == ["Invalid format. Use Name:elem1,elem2"]
    assert ui.plots == []


def test_a_diagram_the_plotter_refuses_to_draw_leaves_the_stats_written(ui, upset_sets, monkeypatch):
    """``plot_upset`` returns None when there is nothing to show; the statistics
    that were already written must stay, and no window may open."""
    monkeypatch.setattr(upset_plots, "plot_upset", lambda *a, **k: None)
    plt.close("all")

    ui.app._run_upset()

    assert ui.text("upset_stats").strip()
    assert ui.plots == []
    assert ui.messages == []


def test_a_core_failure_is_reported_instead_of_an_empty_box(ui, upset_sets, monkeypatch):
    def broken(sets_dict):
        raise RuntimeError("sets must not be empty")

    monkeypatch.setattr(upset_plots, "compute_set_statistics", broken)

    ui.app._run_upset()

    assert ui.messages_of("error") == ["sets must not be empty"]
    assert ui.text("upset_stats").strip() == ""


def test_the_sidebar_entry_shows_the_frame_and_runs_the_analysis(ui, upset_sets):
    ui.app._gui_upset()

    assert ui.app._current_frame == "upset"
    assert ui.text("upset_stats").strip()
    assert ui.plots
