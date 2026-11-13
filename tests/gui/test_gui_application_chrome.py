"""The application chrome: status line, progress, plot history, Ctrl+S and drops.

These are the ``main_window`` helpers every tab leans on, so they are driven here
through the live app rather than through a stub: the status bar is read back from
the label, saved files are opened from disk, and the plot history is inspected for
the figures it holds.  ``_plot_history`` belongs to the session-wide app, so the
fixture saves and restores it.
"""
import sys
import tkinter as tk
import types

import matplotlib.pyplot as plt
import pytest

MISSING = object()


@pytest.fixture
def chrome(ui):
    """The shared app, with its plot history emptied and put back afterwards."""
    app = ui.app
    saved = {name: app.__dict__.get(name, MISSING)
             for name in ("_plot_history", "_plot_history_index", "_progress_bar",
                          "seq_text")}
    app._plot_history = []
    app._plot_history_index = -1
    yield ui
    for name, value in saved.items():
        if value is MISSING:
            app.__dict__.pop(name, None)
        elif name == "seq_text" and not value.winfo_exists():
            # A rebuild inside the test replaced the box with a fresh, live one;
            # putting the destroyed widget back would break every later test.
            continue
        else:
            setattr(app, name, value)


def _fig(title="Figure"):
    fig = plt.figure()
    fig.add_subplot().set_title(title)
    return fig


def test_the_status_line_appends_the_elapsed_time(chrome):
    chrome.app._set_status("Analysis complete", elapsed=12.34)
    assert chrome.status() == "Analysis complete   ·   12.3s"

    chrome.app._set_status("Long run", elapsed=1234.56)
    assert chrome.status() == "Long run   ·   1,234.6s", "thousands are grouped for readability"


def test_a_status_that_cannot_be_written_does_not_break_the_run(chrome, monkeypatch):
    attempted = []

    def refuse(**kwargs):
        attempted.append(kwargs)
        raise tk.TclError("window is gone")

    monkeypatch.setattr(chrome.app.status_bar, "configure", refuse)
    chrome.app._set_status("still alive")

    assert attempted, \
        "the write was never attempted, so the guard had nothing to swallow"


def test_the_right_of_the_status_bar_names_the_active_theme(chrome):
    chrome.app._refresh_status_right()

    shown = chrome.text("status_right").strip()
    assert shown.startswith("theme: "), shown
    assert chrome.app.current_theme_key in shown or chrome.app.T.get('name') in shown, shown


def test_a_theme_label_that_cannot_be_updated_is_ignored(chrome, monkeypatch):
    attempted = []

    def refuse(**kwargs):
        attempted.append(kwargs)
        raise tk.TclError("window is gone")

    monkeypatch.setattr(chrome.app.status_right, "configure", refuse)
    chrome.app._refresh_status_right()

    assert attempted, "the theme label was never touched, so the guard was never tested"


def test_the_working_cursor_is_set_and_cleared(chrome):
    chrome.app._busy_cursor(True)
    assert chrome.app.cget("cursor") == "watch"

    chrome.app._busy_cursor_off()
    assert chrome.app.cget("cursor") == ""


def test_a_cursor_the_window_refuses_to_take_is_not_fatal(chrome, monkeypatch):
    attempted = []

    def refuse(**kwargs):
        attempted.append(kwargs)
        raise tk.TclError("bad cursor")

    monkeypatch.setattr(chrome.app, "configure", refuse)
    chrome.app._busy_cursor(True)

    assert attempted, "no cursor was ever asked for, so the guard proved nothing"


def test_the_progress_bar_tracks_a_value_between_zero_and_one(chrome, monkeypatch):
    seen = []
    chrome.app._show_progress("Rendering 3 of 10...")
    assert chrome.app._progress_bar is not None
    assert chrome.status() == "Rendering 3 of 10..."
    monkeypatch.setattr(chrome.app._progress_bar, "set", seen.append)

    chrome.app._update_progress(0.25)
    chrome.app._update_progress(7)
    chrome.app._update_progress(-3)

    assert seen == [0.25, 1.0, 0.0], "progress must be clamped into the bar's range"

    chrome.app._hide_progress()
    assert chrome.app._progress_bar is None
    assert chrome.status() == "Ready"


def test_hiding_a_progress_bar_that_was_never_shown_is_fine(chrome):
    chrome.app._progress_bar = None

    chrome.app._hide_progress()

    assert chrome.app._progress_bar is None


def test_an_unknown_frame_cannot_be_shown(chrome):
    chrome.app._show_frame('plots')
    before = chrome.app._current_frame

    chrome.app._show_frame('no_such_tab')

    assert chrome.app._current_frame == before


def test_a_frame_still_switches_when_its_title_cannot_be_read(chrome, monkeypatch):
    """Every frame has a sidebar button, so the only way the title lookup fails is
    a window that is going away — and that must not stop the frame switching."""
    assert set(chrome.app.frames) <= set(chrome.app.sidebar_buttons),             "a frame without a button would take the other path"

    def refuse(*args, **kwargs):
        raise tk.TclError("window is gone")

    monkeypatch.setattr(chrome.app.sidebar_buttons['plots'], "cget", refuse)
    chrome.app._show_frame('sequence')

    chrome.app._show_frame('plots')

    assert chrome.app._current_frame == 'plots'


def test_the_window_title_follows_the_sidebar_label(chrome):
    chrome.app._show_frame('plots')

    label = chrome.app.sidebar_buttons['plots'].cget('text').strip()
    assert chrome.app.title().strip() == f"BioSuite Ultra  ·  {label}"


def test_the_plot_history_keeps_ten_names_and_closes_the_eleventh(chrome, monkeypatch):
    figures = [_fig(f"plot {i}") for i in range(11)]
    closed = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close",
                        lambda fig=None: (closed.append(fig), real_close(fig))[1])

    for i, fig in enumerate(figures):
        chrome.app._record_plot(fig, f"plot {i}")

    assert len(chrome.app._plot_history) == 10, "the history must not grow without bound"
    assert [entry['name'] for entry in chrome.app._plot_history] ==             [f"plot {i}" for i in range(1, 11)]
    assert closed == [figures[0]], "the evicted figure must be closed, not just dropped"
    assert chrome.app._plot_history_index == 9


def test_recording_the_same_figure_twice_stores_it_once(chrome):
    fig = _fig("duplicated")

    chrome.app._record_plot(fig, "Volcano Plot")
    chrome.app._record_plot(fig, "Volcano Plot")

    assert len(chrome.app._plot_history) == 1


def test_the_history_starts_over_when_the_app_lacks_one(chrome, monkeypatch):
    monkeypatch.delattr(chrome.app, "_plot_history", raising=False)
    fig = _fig("first")

    chrome.app._record_plot(fig, "First plot")

    assert [entry['name'] for entry in chrome.app._plot_history] == ["First plot"]


def test_the_history_dialog_lists_the_plots_or_says_there_are_none(chrome):
    chrome.app._show_plot_history()
    assert chrome.messages_of("info") == ["No plots generated yet."]

    chrome.app._record_plot(_fig("a"), "Volcano Plot")
    chrome.app._record_plot(_fig("b"), "PCA Plot")
    chrome.app._show_plot_history()

    assert chrome.messages_of("info")[-1] == "1. Volcano Plot\n2. PCA Plot"


def test_ctrl_s_saves_the_sequence_results_box(chrome, tmp_path):
    target = tmp_path / "sequence_results.txt"
    chrome.app._show_frame('sequence')
    chrome.set("seq_stats", "GC: 48.2%\nLength: 120")
    chrome.save_path = str(target)

    chrome.app._save_current()

    assert target.read_text(encoding="utf-8") == "GC: 48.2%\nLength: 120"
    assert chrome.status() == f"Saved: {target}"


def test_ctrl_s_saves_the_latest_figure_at_full_quality(chrome, tmp_path):
    target = tmp_path / "plot.png"
    chrome.app._show_frame('plots')
    chrome.app._record_plot(_fig("Volcano"), "Volcano Plot")
    chrome.save_path = str(target)

    chrome.app._save_current()

    assert target.stat().st_size > 1000, "the figure was not written"
    assert chrome.status() == f"Figure saved: {target}"


def test_ctrl_s_says_so_when_there_is_nothing_to_save(chrome):
    chrome.app._show_frame('sequence')
    chrome.set("seq_stats", "   ")

    chrome.app._save_current()

    assert chrome.status() == "Nothing to save here yet (Ctrl+S)"

    chrome.app._show_frame('help')
    chrome.app._save_current()
    assert chrome.status() == "Nothing to save here yet (Ctrl+S)"


def test_a_cancelled_save_dialog_writes_nothing(chrome, tmp_path):
    chrome.app._show_frame('sequence')
    chrome.set("seq_stats", "data")
    chrome.save_path = ""

    chrome.app._save_current()

    assert list(tmp_path.iterdir()) == []
    assert chrome.messages == []


def test_a_text_save_that_fails_names_the_reason(chrome, tmp_path):
    chrome.set("seq_stats", "data")
    chrome.save_path = str(tmp_path / "absent-dir" / "results.txt")

    chrome.app._save_text_dialog("data", "results.txt")

    errors = chrome.messages_of("error")
    assert len(errors) == 1 and "results.txt" in errors[0], errors


def test_a_figure_that_cannot_be_written_names_the_reason(chrome, tmp_path):
    def refuse(*args, **kwargs):
        raise RuntimeError("renderer has no such format")

    broken = types.SimpleNamespace(savefig=refuse, get_facecolor=lambda: "white")
    chrome.save_path = str(tmp_path / "plot.png")

    chrome.app._save_figure_dialog(broken, "plot")

    assert chrome.messages_of("error") == ["renderer has no such format"]


def test_f5_redraws_the_frame_the_user_is_on(chrome):
    chrome.app._show_frame('cloning')

    chrome.app._refresh_current()

    assert chrome.status() == "Refreshed: cloning"
    assert chrome.app._current_frame == 'cloning'


def test_a_resize_only_schedules_one_layout_pass(chrome, monkeypatch):
    scheduled = []
    monkeypatch.setattr(chrome.app, "after_idle", lambda func=None, *a: scheduled.append(func))
    chrome.app._resize_scheduled = False
    resize = types.SimpleNamespace(widget=chrome.app)

    chrome.app._on_resize(resize)
    chrome.app._on_resize(resize)

    assert len(scheduled) == 1, "the Configure storm must collapse into one idle pass"
    assert chrome.app._resize_scheduled is True


def test_a_resize_of_a_child_widget_is_not_the_window_resizing(chrome, monkeypatch):
    scheduled = []
    monkeypatch.setattr(chrome.app, "after_idle", lambda func=None, *a: scheduled.append(func))

    chrome.app._on_resize(types.SimpleNamespace(widget=chrome.app.status_bar))

    assert scheduled == []


def test_drag_and_drop_is_only_wired_when_the_library_is_there(chrome, monkeypatch):
    """Without tkinterdnd2 the helper stays silent; with it, the widget gets a real
    drop target bound to the handler."""
    registered = {}

    class _DropTarget:
        def drop_target_register(self, kinds):
            registered['kinds'] = kinds

        def dnd_bind(self, event, handler):
            registered['event'] = event
            registered['handler'] = handler

    monkeypatch.setitem(sys.modules, "tkinterdnd2",
                        types.SimpleNamespace(DND_FILES="-dndTypes"))

    chrome.app._setup_drag_drop(_DropTarget())

    assert registered['kinds'] == "-dndTypes"
    assert registered['event'] == "<<Drop>>"
    assert callable(registered['handler'])


def test_a_drop_without_the_dnd_library_wires_nothing(chrome, monkeypatch):
    attempted = []

    class _Refuses:
        def drop_target_register(self, kinds):
            attempted.append(kinds)
            raise AssertionError("must not be called without tkinterdnd2")

    monkeypatch.setitem(sys.modules, "tkinterdnd2", None)

    chrome.app._setup_drag_drop(_Refuses())

    assert attempted == [], "a drop target was registered with the library absent"


def test_a_dropped_file_lands_in_the_widget_that_accepted_it(chrome, tmp_path):
    dropped = tmp_path / "reads.fq"
    dropped.write_text("@r1\nACGT\n+\n####\n", encoding="utf-8")
    chrome.set("seq_text", "old contents")

    chrome.app._on_drop(types.SimpleNamespace(data=f"{{{dropped}}}"), chrome.app.seq_text)

    assert chrome.text("seq_text").rstrip("\n") == str(dropped)
    assert chrome.status() == f"Loaded: {dropped.name}"


def test_a_drop_that_carries_no_file_leaves_the_widget_alone(chrome):
    chrome.set("seq_text", "keep me")

    chrome.app._on_drop(types.SimpleNamespace(data=""), chrome.app.seq_text)

    assert chrome.text("seq_text").rstrip("\n") == "keep me"


def test_a_drop_of_several_files_loads_the_first_and_says_so(chrome, tmp_path):
    first = tmp_path / "first.fq"
    second = tmp_path / "second.fq"
    first.write_text("@r1\nACGT\n+\n####\n", encoding="utf-8")
    second.write_text("@r2\nTGCA\n+\n####\n", encoding="utf-8")
    chrome.set("seq_text", "old contents")

    chrome.app._on_drop(types.SimpleNamespace(data=f"{{{first}}} {{{second}}}"),
                        chrome.app.seq_text)

    assert chrome.text("seq_text").rstrip("\n") == str(first)
    assert chrome.status() == f"Loaded: {first.name} (1 of 2 files)"


def test_rebuilding_stashes_and_restores_the_sequence_entry(chrome):
    """The theme switch destroys every widget; the typed sequence must survive."""
    chrome.set("seq_text", "ACGTACGT")

    chrome.app._rebuild_ui()

    assert chrome.text("seq_text").rstrip("\n") == "ACGTACGT"


def test_rebuilding_survives_an_entry_that_cannot_be_read(chrome, monkeypatch):
    class _Gone:
        def get(self, *args, **kwargs):
            raise tk.TclError("widget is gone")

    chrome.app.seq_text = _Gone()

    chrome.app._rebuild_ui()

    assert chrome.text("seq_text").rstrip("\n") == "", "the fresh box must start empty"


def test_restoring_survives_a_widget_that_cannot_take_text(chrome, monkeypatch):
    attempted = []

    class _Refuses:
        def insert(self, *args, **kwargs):
            attempted.append(args)
            raise tk.TclError("widget is gone")

    chrome.app.seq_text = _Refuses()

    chrome.app._restore_preserved_text({"seq_text": "ACGT"})

    assert attempted == [("1.0", "ACGT")], \
        "the text was never offered to the rebuilt widget"


def test_ctrl_s_with_a_history_entry_that_holds_no_figure_says_so(chrome):
    """A history entry can name a plot without holding its figure; the shortcut must
    report that there is nothing to save rather than fail on None."""
    chrome.app._show_frame('plots')
    chrome.app._plot_history = [{"fig": None, "name": "Volcano Plot"}]

    chrome.app._save_current()

    assert chrome.status() == "Nothing to save here yet (Ctrl+S)"


def test_a_cancelled_figure_dialog_writes_no_file(chrome, tmp_path):
    chrome.save_path = ""

    chrome.app._save_figure_dialog(_fig("Boxplot"), "boxplot")

    assert list(tmp_path.iterdir()) == []
    assert chrome.messages == []


def test_progress_updates_are_ignored_before_the_bar_exists(chrome):
    chrome.app._progress_bar = None

    chrome.app._update_progress(0.5)

    assert chrome.app._progress_bar is None
