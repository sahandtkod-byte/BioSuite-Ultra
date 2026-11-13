"""Theme switching and the start-up fallback.

``_apply_theme`` persists the choice and destroys every frame to rebuild it, so the
config writer is captured here — the user's real settings file is never touched —
and one real rebuild is driven and then undone, so the session app keeps the theme
it was found in.  The last test builds a throwaway app whose stored theme name no
longer exists, which is what a version upgrade leaves behind.
"""
import tkinter as tk

import pytest

from biosuite.core import utils as core_utils
from biosuite.gui import main_window
from biosuite.gui.themes import THEMES


@pytest.fixture
def themes(ui, monkeypatch):
    """The app with its config writer captured and its theme restored."""
    saved_config = dict(core_utils.config)
    written = []
    monkeypatch.setattr(main_window, "save_config", lambda cfg: written.append(dict(cfg)))
    original_key = ui.app.current_theme_key
    yield ui, written
    core_utils.config.clear()
    core_utils.config.update(saved_config)
    if ui.app.current_theme_key != original_key:
        ui.app._apply_theme(original_key)


def _other_theme(app):
    return next(key for key in THEMES if key != app.current_theme_key)


def test_an_unknown_theme_changes_nothing(themes, monkeypatch):
    ui, written = themes
    rebuilt = []
    key = ui.app.current_theme_key
    monkeypatch.setattr(ui.app, "_rebuild_ui", lambda: rebuilt.append(True))

    ui.app._apply_theme('no-such-theme')

    assert ui.app.current_theme_key == key
    assert rebuilt == []
    assert written == []


def test_switching_theme_rebuilds_the_interface_around_the_new_colours(themes):
    """Frames, sidebar, colours and the persisted choice all follow the theme — and
    the text the user had typed must survive the rebuild."""
    ui, written = themes
    target = _other_theme(ui.app)
    before_frames = dict(ui.app.frames)
    before_buttons = dict(ui.app.sidebar_buttons)
    ui.app._show_frame('sequence')
    ui.set("seq_text", "ACGTACGT typed before the switch")

    ui.app._apply_theme(target)

    assert ui.app.current_theme_key == target
    assert ui.app.T is THEMES[target]
    assert written and written[-1]['theme'] == target
    assert core_utils.config['theme'] == target
    assert set(ui.app.frames) == set(before_frames), "a tab frame went missing in the rebuild"
    assert all(ui.app.frames[k] is not before_frames[k] for k in before_frames),             "the frames were never rebuilt"
    assert set(ui.app.sidebar_buttons) == set(before_buttons)
    assert all(ui.app.sidebar_buttons[k] is not before_buttons[k] for k in before_buttons)
    assert ui.text("seq_text").rstrip("\n") == "ACGTACGT typed before the switch",             "the rebuild threw away what the user had typed"
    assert ui.app._current_frame == 'sequence', "the rebuild dropped the user back to another tab"
    assert ui.app._progress_bar is None
    theme_name = THEMES[target].get('name', target)
    assert ui.text("status_right").strip() == f"theme: {theme_name}"
    assert ui.app.cget("fg_color") == THEMES[target]['bg']


def test_a_rebuild_copes_with_widgets_that_are_gone_empty_or_unwritable(themes, monkeypatch):
    """One rebuild exercises all four guards at once: a preserved attribute that no
    longer exists, one whose text cannot be read, one that is blank, and one whose
    new widget refuses the write-back.  A teardown race inside the rebuild must
    never take the window down with it."""
    from biosuite.gui.tabs import sequence_analysis

    ui, _written = themes
    frames_before = set(ui.app.frames)

    class _Refuses:
        def insert(self, *args, **kwargs):
            raise tk.TclError("box is gone")

    original_orf = sequence_analysis.SequenceAnalysisTabMixin._build_orftools_frame

    def build_orf_then_break(self):
        original_orf(self)
        self.orf_result = _Refuses()

    monkeypatch.setattr(type(ui.app), "_PRESERVE_ON_REBUILD",
                        ("seq_text", "gone_widget", "phylo_result", "orf_result"))
    monkeypatch.setattr(sequence_analysis.SequenceAnalysisTabMixin,
                        "_build_orftools_frame", build_orf_then_break)

    def refuse_read(*args, **kwargs):
        raise tk.TclError("widget is being destroyed")

    monkeypatch.setattr(ui.app.seq_text, "get", refuse_read)
    ui.set("phylo_result", "   ")
    ui.set("orf_result", "text that cannot be written back")

    ui.app._rebuild_ui()

    assert set(ui.app.frames) == frames_before, "the rebuild left tabs behind"
    assert ui.app.seq_text.get("1.0", "end-1c") == ""
    assert ui.app._current_frame in ui.app.frames


def test_a_stored_theme_name_that_no_longer_exists_still_starts(monkeypatch):
    """A rename inside THEMES must not stop BioSuite from launching."""
    class _NoSplash:
        def __init__(self, *args, **kwargs):
            pass

    saved_root = tk._default_root
    monkeypatch.setattr(main_window, "BioSplashScreen", _NoSplash)
    monkeypatch.setattr(main_window.BioSuiteApp, "_build_with_splash", lambda self: None)
    monkeypatch.setitem(core_utils.config, 'theme', 'theme-from-an-older-version')

    app = main_window.BioSuiteApp()
    try:
        assert app.current_theme_key in THEMES
        assert app.T is THEMES[app.current_theme_key]
    finally:
        app.destroy()
        tk._default_root = saved_root
