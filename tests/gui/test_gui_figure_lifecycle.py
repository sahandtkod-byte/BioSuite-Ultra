"""Matplotlib figure lifecycle over a long GUI session (finding R5).

``_record_plot`` caps the history at ten figures and closes each evicted one;
a plot window owns its figure and closes it when the user closes the window.
These tests verify the pyplot registry itself — never just a private counter —
and that a display failure leaves neither a window nor a figure behind.
"""
import tkinter as tk

import customtkinter as ctk
import matplotlib.pyplot as plt
import pytest

HISTORY_CAP = 10


@pytest.fixture
def hidden_windows(ui, monkeypatch):
    """The real plot window path, with every window created withdrawn.

    ``ui`` replaces ``_show_plot_from_figure`` with a recorder; this file is
    about figure lifecycles, so the patch is lifted here and the windows are
    hidden instead.
    """
    monkeypatch.delattr(ui.app, "_show_plot_from_figure", raising=False)
    monkeypatch.setattr(ui.app, "_plot_history", [])
    monkeypatch.setattr(ui.app, "_plot_history_index", -1)

    made = []

    class Hidden(ctk.CTkToplevel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.withdraw()
            made.append(self)

    monkeypatch.setattr(ctk, "CTkToplevel", Hidden)
    yield made
    for window in made:
        try:
            window.destroy()
        except tk.TclError:
            pass


def _fig(title):
    fig = plt.figure()
    fig.add_subplot().set_title(title)
    return fig


# ─── The history cap ─────────────────────────────────────────────────────────

def test_recording_more_than_ten_figures_closes_the_evicted_ones(ui):
    figs = [_fig(f"plot {i}") for i in range(HISTORY_CAP + 2)]

    for i, fig in enumerate(figs):
        ui.app._record_plot(fig, f"plot {i}")

    assert len(ui.app._plot_history) == HISTORY_CAP
    for fig in figs[:2]:
        assert not plt.fignum_exists(fig.number), \
            "the evicted figure must be closed, not just dropped from the list"
    for fig in figs[2:]:
        assert plt.fignum_exists(fig.number)
    assert [p["name"] for p in ui.app._plot_history] == \
        [f"plot {i}" for i in range(2, HISTORY_CAP + 2)]
    assert ui.app._plot_history[-1]["fig"] is figs[-1]


def test_a_figure_the_session_does_not_own_survives_the_eviction(ui):
    bystander = _fig("opened outside the recording path")
    figs = [_fig(f"recorded {i}") for i in range(HISTORY_CAP + 1)]

    for i, fig in enumerate(figs):
        ui.app._record_plot(fig, f"recorded {i}")

    assert plt.fignum_exists(bystander.number), \
        "the cap must only ever close figures it was given"
    assert not plt.fignum_exists(figs[0].number)
    assert plt.fignum_exists(figs[-1].number)


def test_the_same_figure_recorded_twice_in_a_row_is_kept_once(ui):
    fig = _fig("duplicate")

    ui.app._record_plot(fig, "duplicate")
    ui.app._record_plot(fig, "duplicate")

    assert len(ui.app._plot_history) == 1
    assert plt.fignum_exists(fig.number), \
        "the dedupe must not close the live figure it refused to store twice"


# ─── A display failure ───────────────────────────────────────────────────────

def test_a_display_failure_leaves_no_window_and_no_figure_behind(
        ui, hidden_windows, monkeypatch):
    """Embedding fails and the static fallback explodes: the user never saw
    anything, so the half-built window and its figure must both be gone."""
    def broken_static(win, fig):
        raise RuntimeError("render exploded")

    def no_embed(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    import matplotlib.backends.backend_tkagg as tkagg

    monkeypatch.setattr(ui.app, "_show_plot_static", broken_static)
    monkeypatch.setattr(tkagg, "FigureCanvasTkAgg", no_embed)
    fig = _fig("doomed")

    with pytest.raises(RuntimeError, match="render exploded"):
        ui.app._show_plot_from_figure(fig, "Doomed")

    assert not plt.fignum_exists(fig.number), \
        "a figure the user never saw must not linger open in pyplot"
    assert hidden_windows and not hidden_windows[0].winfo_exists(), \
        "the half-built window must not outlive the failure"


def test_a_display_failure_whose_cleanup_also_fails_still_raises_the_original(
        ui, monkeypatch):
    """Triple fault behind the R5 cleanup guards: embedding fails, the static
    fallback raises, and BOTH cleanup calls — window.destroy and pyplot.close —
    raise as well.  The guarded cleanup must neither swallow nor mask the original
    error: it still has to surface, and the destroy attempt must still be made.

    The window is a stand-in because it is created inside the method, so its
    destroy can only be made to fail by replacing the constructor.
    """
    from unittest.mock import MagicMock

    import matplotlib.backends.backend_tkagg as tkagg

    monkeypatch.delattr(ui.app, "_show_plot_from_figure", raising=False)
    monkeypatch.setattr(ui.app, "_plot_history", [])
    monkeypatch.setattr(ui.app, "_plot_history_index", -1)

    def broken_static(win, fig):
        raise RuntimeError("render exploded")

    def no_embed(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(ui.app, "_show_plot_static", broken_static)
    monkeypatch.setattr(tkagg, "FigureCanvasTkAgg", no_embed)

    fake_win = MagicMock(name="plot_window")
    fake_win.destroy.side_effect = RuntimeError("destroy failed")
    monkeypatch.setattr(ctk, "CTkToplevel", lambda *a, **k: fake_win)

    fig = _fig("triple fault")
    real_close = plt.close

    def refuse(target=None):
        if target is fig:
            raise RuntimeError("close failed")
        return real_close(target)

    monkeypatch.setattr(plt, "close", refuse)
    try:
        with pytest.raises(RuntimeError, match="render exploded"):
            ui.app._show_plot_from_figure(fig, "Doomed")
        fake_win.destroy.assert_called_once()
    finally:
        real_close(fig)
