"""The Plots Gallery: category selection, the live search box, and rendering.

The gallery is the frame the user lands on, and its handlers do three things the
tests care about: they list ``biosuite.gui.themes.PLOT_CATEGORIES``, they hand the
rendering to a worker thread that posts the figure back with ``after(0, …)``, and
they route every ``input()`` a demo makes into a themed dialog.  ``inline_threads``
runs that worker on the calling thread so a test can assert what the user would
actually see, and every expectation is computed from the same tables the tab reads.
"""
import builtins
import re
import threading
import tkinter as tk
import types

import matplotlib.pyplot as plt
import pytest

from biosuite.gui.tabs import visualization
from biosuite.gui.themes import PLOT_CATEGORIES


@pytest.fixture
def inline_threads(monkeypatch):
    """Replace the tab's ``threading`` with a stub that runs the target inline."""
    class _Inline(threading.Thread):
        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(visualization, "threading",
                        types.SimpleNamespace(Thread=_Inline, Event=threading.Event,
                                              Lock=threading.Lock))
    return _Inline


@pytest.fixture
def gallery(ui):
    """The plots frame in its 'All' state, with the search box empty."""
    ui.app._select_category('All')
    ui.clear("plot_search")
    return ui


def _expected_items(cat=None, term=""):
    """Flatten the category table the way the gallery does."""
    items = []
    if cat is None or cat == 'All':
        for name_cat, plots in PLOT_CATEGORIES.items():
            for name, pid in plots:
                items.append((name, pid, name_cat))
    else:
        for name, pid in PLOT_CATEGORIES[cat]:
            items.append((name, pid, cat))
    if term:
        items = [item for item in items if term in item[0].lower()]
    return items


def test_the_gallery_lists_every_plot_in_the_categories_table(gallery):
    expected = _expected_items()
    assert expected, "the category table is empty, so this test would prove nothing"

    gallery.app._apply_plot_search()

    assert gallery.app._plot_items == expected
    assert gallery.text("plot_count_label") == f"{len(expected)} plots available"


def test_choosing_a_category_lists_only_its_plots(gallery):
    cat = 'Mathematical'
    expected = _expected_items(cat)
    assert 0 < len(expected) < len(_expected_items())

    gallery.app._select_category(cat)

    assert gallery.app._plot_items == expected
    assert gallery.text("plot_count_label") == f"{len(expected)} plots available"
    T = gallery.app.T
    assert gallery.app.cat_buttons[cat].cget('fg_color') == T['sidebar_active']
    for other, btn in gallery.app.cat_buttons.items():
        if other != cat:
            assert btn.cget('fg_color') == 'transparent', f"{other} is still highlighted"


def test_typing_in_the_search_box_filters_the_list_by_name(gallery):
    term = "plot"
    gallery.set("plot_search", term)

    gallery.app._on_search_key()

    expected = _expected_items('All', term)
    assert expected, f"no plot name contains {term!r}; pick a term the table has"
    assert gallery.app._plot_items == expected
    assert gallery.text("plot_count_label") == f"{len(expected)} plots available"


def test_a_second_keystroke_cancels_the_update_still_queued(gallery, monkeypatch):
    """An unmatched term must leave an empty list, and the earlier queued rebuild
    must not fire on top of it."""
    gallery.set("plot_search", "no-such-plot-name")
    gallery.app._on_search_key()
    assert gallery.app._plot_items == []
    pending = gallery.app._search_after_id
    assert pending, "after() handed back no id for the debounced update"

    cancelled = []
    monkeypatch.setattr(gallery.app, "after_cancel",
                        lambda after_id: cancelled.append(after_id))
    gallery.set("plot_search", "heat")
    gallery.app._on_search_key()

    assert cancelled == [pending]
    assert gallery.app._plot_items == _expected_items('All', "heat")


def test_the_clear_button_empties_the_box_and_restores_the_list(gallery):
    gallery.set("plot_search", "venn")
    gallery.app._on_search_key()
    assert gallery.app._plot_items == _expected_items('All', "venn")

    gallery.app._clear_plot_search()

    assert gallery.text("plot_search") == ""
    assert gallery.app._plot_items == _expected_items()
    assert gallery.app._search_after_id is None, "the queued update was left pending"


def test_clearing_the_box_survives_a_cancel_that_is_refused(gallery, monkeypatch):
    gallery.app._search_after_id = 4242
    monkeypatch.setattr(gallery.app, "after_cancel",
                        lambda after_id: (_ for _ in ()).throw(tk.TclError("bad after id")))

    gallery.app._clear_plot_search()

    assert gallery.app._plot_items == _expected_items()
    assert gallery.app._search_after_id is None


def test_a_second_keystroke_survives_a_stale_debounce_id(gallery, monkeypatch):
    """A 150 ms debounce that already fired leaves a stale id behind; cancelling it
    must not raise into the keystroke."""
    gallery.app._search_after_id = 4242
    monkeypatch.setattr(gallery.app, "after_cancel",
                        lambda after_id: (_ for _ in ()).throw(tk.TclError("bad after id")))
    gallery.set("plot_search", "heat")

    gallery.app._on_search_key()

    assert gallery.app._search_after_id, "the new debounce was never queued"
    assert gallery.app._plot_items == _expected_items('All', "heat")


def test_the_search_is_harmless_before_the_gallery_frame_exists(gallery, monkeypatch):
    monkeypatch.delattr(gallery.app, "plot_buttons_frame")
    before = gallery.text("plot_count_label")

    gallery.app._apply_plot_search()

    assert gallery.text("plot_count_label") == before


def test_the_list_falls_back_to_every_plot_when_the_search_box_is_gone(gallery, monkeypatch):
    monkeypatch.delattr(gallery.app, "plot_search")

    gallery.app._apply_plot_search()

    assert gallery.app._plot_items == _expected_items()


def test_generating_without_a_selection_asks_for_one(gallery):
    gallery.app._selected_plot_id = None

    gallery.app._generate_selected_plot()

    assert gallery.messages_of("info") == ["Select a plot from the list, then click Generate."]
    assert gallery.plots == []


def test_generating_a_plot_displays_the_figure_that_plot_drew(gallery, inline_threads):
    plt.close("all")
    gallery.confirm_answer = False
    gallery.app._selected_plot_id = "sine"

    gallery.app._generate_selected_plot()

    assert gallery.plots,             "the gallery never displayed the plot the user asked for"
    title, fig = gallery.plots[-1]
    assert title == "Sine"
    drawn = " ".join(ax.get_title() for ax in fig.axes)
    assert re.search(r"Sine", drawn), f"the window was handed a different figure: {drawn!r}"
    assert gallery.status() == "Ready"
    assert gallery.messages == []


def _row_button(app, name):
    """The list button labelled ``name``, or None."""
    for row in app.plot_buttons_frame.winfo_children():
        for child in row.winfo_children():
            if hasattr(child, "cget") and child.cget("text") == f"  {name}":
                return child
    return None


def test_clicking_a_row_selects_that_plot_and_renders_it(gallery, inline_threads):
    """The rows are built with a lambda each; a click must remember the choice and
    draw it, or Generate and the list disagree about what the user picked."""
    plt.close("all")
    gallery.confirm_answer = False
    button = _row_button(gallery.app, "Sine")
    assert button is not None, "the Sine row has no button to click"

    button.invoke()

    assert gallery.app._selected_plot_id == "sine"
    assert gallery.plots, "clicking the row did not display the plot"
    assert gallery.plots[-1][0] == "Sine"
    assert gallery.status() == "Ready"


def test_clearing_the_box_with_nothing_queued_still_refreshes_the_list(gallery):
    gallery.app._search_after_id = None

    gallery.app._clear_plot_search()

    assert gallery.app._plot_items == _expected_items()
    assert gallery.app._search_after_id is None


def test_clearing_the_box_survives_an_entry_that_refuses_to_clear(gallery, monkeypatch):
    """The guard swallows the failed clear and rebuilds from what the box still
    holds, so the click cannot take the whole gallery down."""
    def refuse(*args):
        raise tk.TclError("entry is gone")

    gallery.set("plot_search", "venn")
    gallery.app._on_search_key()
    pending = gallery.app._search_after_id
    monkeypatch.setattr(gallery.app.plot_search, "delete", refuse)

    gallery.app._clear_plot_search()

    assert gallery.app._search_after_id is None
    assert gallery.app._plot_items == _expected_items('All', "venn"),         "the list was rebuilt from the text the box kept"
    assert pending, "the queued update was never taken out"


def test_a_demo_that_raises_is_reported_and_leaves_nothing_behind(
        gallery, inline_threads, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("amplitude is not a number")

    monkeypatch.setattr(visualization, "PLOT_FUNCS", {"broken": explode})
    original_input, original_show = builtins.input, plt.show

    gallery.app._generate_plot_by_id("broken")

    assert gallery.messages_of("error") == ["amplitude is not a number"]
    assert builtins.input is original_input, "the demo's input patch was left in place"
    assert plt.show is original_show, "plt.show was left stubbed for the whole process"
    assert gallery.status() == "Ready"
    assert gallery.app._progress_bar is None, "the progress bar stayed up after the failure"


def test_an_unknown_plot_id_is_named_instead_of_rendering_nothing(gallery, inline_threads, monkeypatch):
    monkeypatch.setattr(visualization, "PLOT_FUNCS", {})

    gallery.app._generate_plot_by_id("no_such_plot")

    assert gallery.messages_of("error") == ["Plot 'no_such_plot' not found."]
    assert gallery.plots == []


def test_the_progress_bar_drops_when_the_render_finishes(gallery, inline_threads, monkeypatch):
    shown = []
    monkeypatch.setattr(visualization, "PLOT_FUNCS", {"sine": lambda *a, **k: None})
    real_show_progress = gallery.app._show_progress
    monkeypatch.setattr(gallery.app, "_show_progress",
                        lambda text="Working...": (shown.append(text), real_show_progress(text))[1])

    gallery.app._generate_plot_by_id("sine")

    assert shown == ["Generating sine..."]
    assert gallery.app._progress_bar is None, "the gallery left a progress bar packed"


def test_plot_names_are_taken_from_the_category_table(gallery):
    for _cat, plots in PLOT_CATEGORIES.items():
        for name, pid in plots:
            assert gallery.app._plot_name_for(pid) == name

    assert gallery.app._plot_name_for("custom_thing") == "Custom Thing"


def test_the_prompt_router_uses_a_yes_no_dialog_for_confirmation_questions(gallery):
    gallery.confirm_answer = True
    assert gallery.app._gui_input("Load data from file?") == 'y'
    assert gallery.last_question() == ("confirm", "Load data from file?")

    gallery.confirm_answer = False
    for prompt in ("Load ranked list from file?", "Show data summary", "Use default",
                   "Save this plot? (y/n): ", "Save as HTML?"):
        assert gallery.app._gui_input(prompt) == 'n', prompt


def test_the_prompt_router_asks_for_a_file_when_the_prompt_wants_a_path(gallery, tmp_path):
    gallery.input_answer = None
    gallery.open_path = ""

    assert gallery.app._gui_input("File path of the CSV?") == ""
    assert gallery.last_question() == ("file", "File path of the CSV?")

    chosen = tmp_path / "counts.csv"
    gallery.open_path = str(chosen)
    assert gallery.app._gui_input("Enter the file path") == str(chosen)


def test_the_prompt_router_asks_for_a_column_by_name(gallery):
    gallery.input_answer = "log2FC"

    assert gallery.app._gui_input("Which column holds the values?") == "log2FC"
    assert gallery.last_question() == ("input", "Which column holds the values?")


def test_the_prompt_router_passes_a_shown_default_through(gallery):
    gallery.input_answer = "2.5"
    assert gallery.app._gui_input("Amplitude A (default 1): ") == "2.5"
    assert gallery.last_question() == ("input", "Amplitude A (default 1): ")

    gallery.input_answer = None
    assert gallery.app._gui_input("Amplitude A (default 1): ") == "1",             "a dismissed dialog must fall back to the default it showed"


def test_the_prompt_router_keeps_these_questions_silent(gallery):
    gallery.input_answer = "typed"

    for prompt in ("Enter sequences (FASTA): ", "Switch to dark mode?",
                   "Filename (default sine.png): "):
        assert gallery.app._gui_input(prompt) == "", prompt
    assert gallery.asked == []


def test_the_prompt_router_answers_a_correlation_question(gallery):
    gallery.input_answer = None
    assert gallery.app._gui_input("Correlation type (pearson/spearman): ") == "pearson"


def test_exporting_all_plots_writes_the_batch_into_the_chosen_folder(
        gallery, inline_threads, monkeypatch, tmp_path):
    from biosuite.plotting import biological_plots
    from biosuite.plotting.math_plots import cosine_plot, sine_plot

    monkeypatch.setattr(biological_plots, "_batch_plot_functions",
                        lambda: [sine_plot, cosine_plot])
    folder = tmp_path / "gallery_export"
    gallery.directory = str(folder)

    gallery.app._export_all_plots()

    written = sorted(p.name for p in folder.glob("*.png"))
    assert written == ["cosine.png", "sine.png"], f"the export wrote {written}"
    assert gallery.messages_of("success") == [f"All plots exported to:\n{folder}"]
    assert gallery.status() == "Ready"


def test_export_reports_a_folder_it_could_not_write(gallery, inline_threads, monkeypatch, tmp_path):
    def refuse(folder):
        raise OSError("disk full")

    monkeypatch.setattr(gallery.app, "_export_all_to_folder", refuse)
    gallery.directory = str(tmp_path / "nowhere")

    gallery.app._export_all_plots()

    assert gallery.messages_of("error") == ["disk full"]
    assert gallery.status() == "Ready"


def test_cancelling_the_export_dialog_exports_nothing(gallery, inline_threads, monkeypatch):
    called = []
    monkeypatch.setattr(gallery.app, "_export_all_to_folder",
                        lambda folder: called.append(folder))
    gallery.directory = ""
    before = gallery.status()

    gallery.app._export_all_plots()

    assert called == []
    assert gallery.status() == before


def test_the_batch_pdf_is_written_where_the_dialog_said(gallery, inline_threads, monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(gallery.app, "_batch_export_to_pdf",
                        lambda path: written.append(path))
    target = tmp_path / "report.pdf"
    gallery.save_path = str(target)

    gallery.app._batch_pdf()

    assert written == [str(target)]
    assert gallery.messages_of("success") == [f"PDF saved:\n{target}"]
    assert gallery.status() == "Ready"


def test_a_pdf_failure_is_reported_with_its_reason(gallery, inline_threads, monkeypatch, tmp_path):
    def refuse(path):
        raise RuntimeError("no pages to write")

    monkeypatch.setattr(gallery.app, "_batch_export_to_pdf", refuse)
    gallery.save_path = str(tmp_path / "report.pdf")

    gallery.app._batch_pdf()

    assert gallery.messages_of("error") == ["no pages to write"]
    assert gallery.status() == "Ready"


def test_cancelling_the_pdf_dialog_writes_nothing(gallery, inline_threads, monkeypatch):
    written = []
    monkeypatch.setattr(gallery.app, "_batch_export_to_pdf", lambda path: written.append(path))
    gallery.save_path = ""
    before = gallery.status()

    gallery.app._batch_pdf()

    assert written == []
    assert gallery.status() == before
