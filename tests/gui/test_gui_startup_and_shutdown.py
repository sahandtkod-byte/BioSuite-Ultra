"""Startup and shutdown: what the app does when the window is closed or a callback fails.

``_on_close`` is wired to the title-bar X and to Ctrl+Q, and
``report_callback_exception`` decides whether a Tk callback error is printed or
swallowed.  Both are driven here on stand-in objects rather than on the session
app, because closing the real window would end every other GUI test.
"""
import tkinter as tk

import pytest

from biosuite.gui import main_window


class _Window:
    """Just enough of the app for the close and error-reporting paths."""

    def __init__(self, closing=False, destroy_error=None):
        self._closing = closing
        self.destroyed = False
        self._destroy_error = destroy_error

    def destroy(self):
        if self._destroy_error:
            raise self._destroy_error
        self.destroyed = True


def test_closing_the_window_closes_every_figure_and_destroys_the_app(monkeypatch):
    closed = []
    monkeypatch.setattr(main_window, "plt",
                        type("FakePyplot", (), {"close": staticmethod(
                            lambda kind: closed.append(kind))}))
    window = _Window()

    main_window.BioSuiteApp._on_close(window)

    assert closed == ["all"], "the figures were not released on shutdown"
    assert window.destroyed
    assert window._closing is True, "the flag that silences teardown noise was never set"


def test_a_figure_library_that_is_not_loaded_still_shuts_down(monkeypatch):
    monkeypatch.setattr(main_window, "plt", None)
    window = _Window()

    main_window.BioSuiteApp._on_close(window)

    assert window.destroyed


def test_a_destroy_that_fails_at_shutdown_is_swallowed(monkeypatch):
    """By the time the user hits X the Tk window may already be gone; a second
    destroy must not raise into the callback that is trying to quit the app."""
    monkeypatch.setattr(main_window, "plt", None)
    window = _Window(destroy_error=tk.TclError("window already gone"))

    main_window.BioSuiteApp._on_close(window)

    assert not window.destroyed


def test_a_figure_library_that_refuses_to_close_does_not_stop_the_shutdown(monkeypatch):
    class _BrokenPyplot:
        @staticmethod
        def close(kind):
            raise RuntimeError("no display to close")

    window = _Window()
    monkeypatch.setattr(main_window, "plt", _BrokenPyplot())

    main_window.BioSuiteApp._on_close(window)

    assert window.destroyed, "the app must still go away if the figures refuse"


def test_other_tcl_errors_are_printed_even_while_closing(capsys):
    """Only the 'invalid command name' teardown noise is suppressed; a real Tcl
    failure has to reach the console."""
    window = _Window(closing=True)
    error = tk.TclError("bad window path name \".!biosuiteapp\"")

    main_window.BioSuiteApp.report_callback_exception(window, type(error), error, None)

    assert "bad window path name" in capsys.readouterr().err


def test_teardown_noise_is_swallowed_only_while_closing(capsys):
    window = _Window(closing=True)
    error = tk.TclError("invalid command name \"140234567890\"")

    main_window.BioSuiteApp.report_callback_exception(window, type(error), error, None)

    assert capsys.readouterr().out == "", "a dying callback should not print"


def test_any_other_callback_error_is_printed_in_full(capsys):
    window = _Window(closing=True)
    error = ValueError("a tab did something wrong")

    main_window.BioSuiteApp.report_callback_exception(window, type(error), error, None)

    printed = capsys.readouterr().err
    assert "Exception in Tkinter callback" in printed
    assert "a tab did something wrong" in printed


def test_a_callback_error_is_printed_when_the_app_is_not_closing(capsys):
    window = _Window(closing=False)
    error = tk.TclError("invalid command name \"140234567890\"")

    main_window.BioSuiteApp.report_callback_exception(window, type(error), error, None)

    assert "invalid command name" in capsys.readouterr().err


def test_the_startup_sequence_shows_the_window_after_sizing_it():
    """The window must be sized before it is mapped, or it appears in a corner and
    then jumps."""
    class _Shown(_Window):
        def __init__(self):
            super().__init__()
            self.calls = []

        def deiconify(self):
            self.calls.append("deiconify")

        def update_idletasks(self):
            self.calls.append("idle")

        def _auto_size_and_center(self):
            self.calls.append("center")

        def lift(self):
            self.calls.append("lift")

        def focus_force(self):
            self.calls.append("focus")

    window = _Shown()

    main_window.BioSuiteApp._show_main_window(window)

    assert window.calls == ["deiconify", "idle", "center", "lift", "focus"]
