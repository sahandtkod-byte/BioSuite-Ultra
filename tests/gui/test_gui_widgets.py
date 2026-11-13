"""The themed tooltip: it waits, it opens once, and it goes away with the pointer.

The app window is never shown, so Tk delivers no <Enter>/<Leave> events to its
widgets.  These tests therefore call the very methods those bindings are wired to,
on a real button in the hidden app window, with the button's own ``after`` running
the scheduled callback on the spot.  Everything is read back from the tip window
the Tooltip builds, or from the geometry Tk was asked to place it at.
"""
import customtkinter as ctk
import pytest
import tkinter as tk

from biosuite.gui.themes import THEMES
from biosuite.gui.widgets import Tooltip, attach_tooltip

PURPLE = THEMES["dark-purple"]
HINT = "Scores the genes that changed the most"


def _placed_at(window):
    return tuple(int(v) for v in window.geometry().rsplit("+", 2)[1:])


def _peer_is_gone(*args):
    raise tk.TclError("bad window path name")


class Host:
    """A real button, a way to hang tooltips on it, and a wait that never waits."""

    def __init__(self, app):
        self.app = app
        self.button = ctk.CTkButton(app, text="host")
        self.button.pack()
        self.buttons = [self.button]
        self.tips = []
        self.waited = []

    def spare_button(self):
        button = ctk.CTkButton(self.app, text="spare")
        button.pack()
        self.buttons.append(button)
        return button

    def attach(self, text=HINT, theme=PURPLE, delay=None, widget=None):
        tip = Tooltip(widget if widget is not None else self.button, text, theme, delay)
        self.tips.append(tip)
        return tip

    def never_waits(self):
        def run_now(milliseconds, func=None, *args):
            self.waited.append(milliseconds)
            if func is not None:
                func(*args)
            return "never-really-pending"

        self.button.after = run_now

    def shown_label(self, tip):
        return tip._tip.winfo_children()[0].winfo_children()[0]

    def clean_up(self):
        for tip in self.tips:
            tip._hide()
        for button in reversed(self.buttons):
            try:
                button.destroy()
            except tk.TclError:
                pass


@pytest.fixture
def host(gui_app):
    handled = Host(gui_app)
    yield handled
    handled.clean_up()


def test_a_hover_asks_for_the_tip_after_the_declared_pause(host):
    host.never_waits()
    tip = host.attach(delay=17)

    tip._schedule()

    assert host.waited == [17]
    assert tip._tip is not None


def test_the_default_pause_comes_from_the_class_constant(host):
    host.never_waits()
    tip = host.attach()

    tip._schedule()

    assert host.waited == [Tooltip.DELAY_MS]
    assert tip._tip is not None


def test_the_tip_window_shows_the_hint_in_the_theme_colours(host):
    host.never_waits()
    tip = host.attach()

    tip._schedule()

    frame = tip._tip.winfo_children()[0]
    label = frame.winfo_children()[0]
    assert label.cget("text") == HINT
    assert label.cget("bg") == PURPLE["card"]
    assert label.cget("fg") == PURPLE["text"]
    assert frame.cget("bg") == PURPLE["border_light"]
    assert frame.cget("bd") == 0
    assert int(label.cget("padx")) == Tooltip.PAD_X


def test_a_hint_without_a_theme_still_opens_a_readable_window(host):
    host.never_waits()
    themed = host.attach()
    themed._schedule()

    plain = host.attach(theme=None)
    plain._schedule()

    assert plain._tip is not None
    assert host.shown_label(plain).cget("text") == HINT
    assert host.shown_label(plain).cget("bg") != host.shown_label(themed).cget("bg")


def test_the_tip_opens_below_the_widget_it_explains(host):
    host.never_waits()
    host.button.winfo_rootx = lambda: 100
    host.button.winfo_rooty = lambda: 50
    tip = host.attach()

    tip._schedule()

    expected_y = 50 + host.button.winfo_height() + Tooltip.OFFSET_Y
    assert _placed_at(tip._tip) == (100, expected_y)


def test_a_tip_that_would_overrun_the_screen_edge_is_pulled_back(host):
    host.never_waits()
    tip = host.attach()
    host.button.winfo_rootx = lambda: 100
    tip._schedule()
    fits_at = _placed_at(tip._tip)[0]
    tip._hide()

    host.button.winfo_rootx = lambda: host.button.winfo_screenwidth() + 400
    tip._schedule()

    window = tip._tip
    pulled_back_to = window.winfo_screenwidth() - window.winfo_reqwidth() - 8
    assert _placed_at(window)[0] == pulled_back_to
    assert fits_at == 100, "a tip that fits keeps the widget's own left edge"
    assert pulled_back_to < window.winfo_screenwidth()


def test_a_widget_at_the_left_edge_of_the_screen_still_gets_a_visible_tip(host):
    host.never_waits()
    host.button.winfo_rootx = lambda: -400
    tip = host.attach()

    tip._schedule()

    assert _placed_at(tip._tip)[0] == 0


def test_a_tip_that_would_overrun_the_bottom_is_pulled_up(host):
    host.never_waits()
    tip = host.attach()
    host.button.winfo_rooty = lambda: host.button.winfo_screenheight() - 5

    tip._schedule()

    window = tip._tip
    expected_y = window.winfo_screenheight() - window.winfo_reqheight() - 8
    assert _placed_at(window)[1] == expected_y
    assert expected_y >= 0


def test_a_widget_whose_timer_cannot_start_leaves_no_pending_tip(host, monkeypatch):
    tip = host.attach()
    monkeypatch.setattr(host.button, "after", _peer_is_gone, raising=False)

    tip._schedule()

    assert tip._after_id is None
    assert tip._tip is None


def test_the_tip_opens_only_once_however_often_the_widget_is_entered(host):
    host.never_waits()
    tip = host.attach()

    tip._schedule()
    window = tip._tip
    tip._schedule()
    tip._schedule()

    assert tip._tip is window
    opened = [c for c in host.button.winfo_children() if isinstance(c, tk.Toplevel)]
    assert len(opened) == 1


def test_leaving_the_widget_destroys_the_tip_and_clears_the_wait(host):
    host.never_waits()
    tip = host.attach()
    tip._schedule()
    window = tip._tip
    assert window.winfo_exists()

    tip._hide()

    assert tip._tip is None
    assert not window.winfo_exists()
    assert tip._after_id is None


def test_a_press_cancels_the_wait_before_the_tip_ever_opens(host):
    tip = host.attach()

    tip._schedule()

    assert tip._after_id is not None
    assert tip._tip is None, "the tip must not appear before its delay elapses"

    tip._show()
    window = tip._tip
    assert window is not None and window.winfo_exists()

    tip._hide()

    assert tip._after_id is None and tip._tip is None
    assert not window.winfo_exists()


def test_a_widget_destroyed_while_the_tip_waits_leaves_no_window(host):
    button = host.spare_button()
    tip = host.attach(widget=button)
    tip._schedule()

    button.destroy()
    tip._show()
    tip._hide()

    assert tip._tip is None, "a dead widget must not get a tip"
    assert tip._after_id is None


def test_the_tip_stays_closed_when_its_widget_cannot_answer_at_all(host):
    """Tk can drop a widget's peer between the hover and the timer firing; the tip
    must then stay shut instead of raising out of the event loop."""
    host.never_waits()
    alive = host.attach()
    alive._schedule()
    assert alive._tip is not None

    host.button.winfo_exists = _peer_is_gone
    waiting = host.attach()
    waiting._schedule()

    assert waiting._tip is None


def test_a_tip_window_that_refuses_to_close_is_still_dropped(host, monkeypatch):
    host.never_waits()
    tip = host.attach()
    tip._schedule()
    window = tip._tip
    assert window.winfo_exists()
    monkeypatch.setattr(window, "destroy", _peer_is_gone, raising=False)

    tip._hide()

    assert tip._tip is None
    assert window.winfo_exists(), "only the guard ran, the window was never closed"


def test_a_widget_whose_timer_cannot_be_cancelled_still_loses_its_tip(host, monkeypatch):
    """Cancelling the pending wait can fail once the widget's Tk peer is gone; the
    hide path has to swallow that and take the tip down anyway."""
    host.never_waits()
    tip = host.attach()
    tip._schedule()
    window = tip._tip
    assert window.winfo_exists()
    monkeypatch.setattr(host.button, "after_cancel", _peer_is_gone, raising=False)

    tip._hide()

    assert tip._tip is None
    assert tip._after_id is None
    assert not window.winfo_exists()


def test_an_empty_hint_leaves_the_widget_untouched(host):
    assert attach_tooltip(host.button, "", PURPLE) is None

    attached = attach_tooltip(host.button, HINT, PURPLE)
    assert attached is not None
    host.never_waits()
    attached._schedule()
    assert host.shown_label(attached).cget("text") == HINT
    attached._hide()


def test_a_hint_on_a_widget_that_cannot_be_bound_is_dropped(host):
    gone = host.spare_button()
    gone.destroy()

    assert attach_tooltip(gone, HINT, PURPLE) is None

    attached = attach_tooltip(host.button, HINT, PURPLE)
    assert attached is not None
    host.never_waits()
    attached._schedule()
    assert host.shown_label(attached).cget("text") == HINT
