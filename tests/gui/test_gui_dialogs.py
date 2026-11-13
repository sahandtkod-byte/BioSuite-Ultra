"""The themed dialogs the app's blocking helpers build: message, confirm, input, file.

``ui`` cannot be used here — that fixture exists to disarm exactly these classes —
so every test builds a real dialog on the hidden app window, clicks the button the
dialog itself renders (``invoke()``) or asks Tk to run the close handler the dialog
registered, and reads the answer back from ``result`` plus the widgets it built.
"""
import customtkinter as ctk
import pytest
import tkinter.filedialog

from biosuite.gui.dialogs import (
    BioConfirmDialog,
    BioDropdownDialog,
    BioFilePickerDialog,
    BioInputDialog,
    BioMessageDialog,
    BioSplashScreen,
)
from biosuite.gui.themes import THEMES

PURPLE = THEMES["dark-purple"]
SHORT = "Run the analysis now?"
LONG = "Re-running this analysis overwrites the report that is already sitting in the box"
SIZES = [
    (BioConfirmDialog, dict(message=SHORT), 440, 180),
    (BioInputDialog, dict(prompt="name"), 460, 230),
    (BioFilePickerDialog, dict(prompt="pick"), 500, 200),
    (BioDropdownDialog, dict(prompt="pick", options=["a"]), 440, 250),
]


def _widgets(dialog, cls):
    found = []

    def walk(widget):
        for child in widget.winfo_children():
            if isinstance(child, cls):
                found.append(child)
            walk(child)

    walk(dialog)
    return found


def _button(dialog, label):
    matches = [b for b in _widgets(dialog, ctk.CTkButton) if b.cget("text") == label]
    assert len(matches) == 1, "the dialog should render exactly one such button"
    return matches[0]


def _label_texts(dialog):
    return [label.cget("text") for label in _widgets(dialog, ctk.CTkLabel)]


def _placed_at(window):
    return tuple(int(v) for v in window.geometry().rsplit("+", 2)[1:])


def _grab_is_free(window):
    return window.tk.eval("grab current") == ""


def _refuse(*args):
    raise RuntimeError("the dialog does not hold the grab")


@pytest.fixture
def dialog(gui_app):
    built = []

    def make(cls, theme=PURPLE, **kwargs):
        window = cls(gui_app, theme, **kwargs)
        built.append(window)
        return window

    yield make
    for window in built:
        try:
            window.grab_release()
        except Exception:
            pass
        try:
            window.destroy()
        except Exception:
            pass


def test_a_dialog_is_painted_from_the_theme_it_is_given(dialog, gui_app):
    themed = dialog(BioConfirmDialog, message=SHORT)

    assert themed.cget("fg_color") == PURPLE["bg"]
    assert themed._outer.cget("fg_color") == PURPLE["dialog_border"]
    assert themed._card.cget("fg_color") == PURPLE["dialog_bg"]
    assert themed._body.cget("fg_color") == "transparent"

    from_app = dialog(BioConfirmDialog, theme=gui_app.T, message=SHORT)
    assert from_app.cget("fg_color") == gui_app.T["bg"]


def test_a_theme_without_dialog_colours_falls_back_to_the_border_and_card(dialog):
    plain = {k: v for k, v in PURPLE.items() if k not in ("dialog_border", "dialog_bg")}
    assert plain["border"] != PURPLE["dialog_border"]
    assert plain["card"] != PURPLE["dialog_bg"]

    d = dialog(BioConfirmDialog, theme=plain, message=SHORT)

    assert d._outer.cget("fg_color") == plain["border"]
    assert d._card.cget("fg_color") == plain["card"]


def test_every_dialog_centres_its_own_size_on_the_screen(dialog):
    for cls, kwargs, width, height in SIZES:
        d = dialog(cls, **kwargs)
        left, top = _placed_at(d)
        assert left == (d.winfo_screenwidth() - width) // 2, cls.__name__
        assert top == (d.winfo_screenheight() - height) // 2, cls.__name__
        d.destroy()


def test_a_long_message_gets_the_taller_dialog_than_a_short_one(dialog):
    short = dialog(BioMessageDialog, message=SHORT, msg_type="info")
    edge = _placed_at(short)[1]
    assert edge == (short.winfo_screenheight() - 180) // 2
    short.destroy()

    tall = dialog(BioMessageDialog, message=LONG, msg_type="info")

    assert _placed_at(tall)[1] == (tall.winfo_screenheight() - 220) // 2
    assert _placed_at(tall)[1] == edge - 20


def test_a_very_long_message_grows_the_dialog_instead_of_clipping_it(dialog):
    d = dialog(BioMessageDialog, message="x" * 1000, msg_type="error")

    assert d._fitted_height > 220, "the fixed 220px dialog would clip this message"
    assert d._fitted_height >= d.winfo_reqheight(), "the content still asks for more room"
    assert _placed_at(d)[1] == (d.winfo_screenheight() - d._fitted_height) // 2


def test_closing_a_message_dialog_from_its_title_bar_answers_nothing(dialog):
    """BioMessageDialog is the one class that keeps the base close handler: the
    answer stays undecided rather than becoming an acknowledgement."""
    d = dialog(BioMessageDialog, message=SHORT, msg_type="info")

    d.tk.eval(d.protocol("WM_DELETE_WINDOW"))

    assert not d.winfo_exists()
    assert d.result is None


def test_an_open_dialog_holds_the_grab_until_it_is_answered(dialog, gui_app):
    assert _grab_is_free(gui_app)
    d = dialog(BioConfirmDialog, message=SHORT)

    d._do_grab()
    assert gui_app.tk.eval("grab current") == d._w

    _button(d, "Yes").invoke()
    assert _grab_is_free(gui_app)


def test_a_dialog_still_answers_when_the_grab_is_refused(dialog, monkeypatch):
    d = dialog(BioConfirmDialog, message=SHORT)
    monkeypatch.setattr(d, "grab_set", _refuse, raising=False)

    d._do_grab()

    assert _grab_is_free(d)
    _button(d, "Yes").invoke()
    assert d.result is True


def test_the_yes_button_answers_true_and_the_cancel_button_false(dialog):
    yes = dialog(BioConfirmDialog, message=SHORT)
    _button(yes, "Yes").invoke()
    assert yes.result is True
    assert not yes.winfo_exists()

    no = dialog(BioConfirmDialog, message=SHORT)
    _button(no, "Cancel").invoke()

    assert no.result is False
    assert not no.winfo_exists()


def test_the_message_dialog_shows_the_text_it_was_asked_to(dialog):
    d = dialog(BioMessageDialog, message=SHORT, msg_type="warning")

    assert SHORT in _label_texts(d)
    assert _button(d, "OK").cget("fg_color") == PURPLE["accent"]


def test_each_message_kind_carries_its_own_icon_and_colour(dialog):
    shown = {}
    for kind in ("info", "warning", "error", "success"):
        d = dialog(BioMessageDialog, message=SHORT, msg_type=kind)
        icon = _widgets(d, ctk.CTkLabel)[0]
        shown[kind] = (icon.cget("text"), icon.cget("text_color"))
        d.destroy()

    assert len({icon for icon, _ in shown.values()}) == 4
    assert len({colour for _, colour in shown.values()}) == 4
    assert shown["info"][0] != shown["warning"][0]


def test_an_unknown_message_kind_is_shown_as_info(dialog):
    info = dialog(BioMessageDialog, message=SHORT, msg_type="info")
    unknown = dialog(BioMessageDialog, message=SHORT, msg_type="blessing")
    info_icon = _widgets(info, ctk.CTkLabel)[0]
    unknown_icon = _widgets(unknown, ctk.CTkLabel)[0]

    assert unknown_icon.cget("text") == info_icon.cget("text")
    assert unknown_icon.cget("text_color") == info_icon.cget("text_color")
    assert unknown_icon.cget("text_color") == PURPLE["accent"]


def test_an_error_dialog_writes_its_button_in_a_light_ink(dialog):
    error = dialog(BioMessageDialog, message=SHORT, msg_type="error")
    info = dialog(BioMessageDialog, message=SHORT, msg_type="info")

    assert (_button(info, "OK").cget("text_color"),
            _button(error, "OK").cget("text_color")) == ("#000000", "#ffffff")


def test_ok_is_the_only_answer_a_message_dialog_gives(dialog):
    d = dialog(BioMessageDialog, message=SHORT, msg_type="success")

    _button(d, "OK").invoke()

    assert d.result == "ok"
    assert not d.winfo_exists()


def test_the_input_dialog_opens_on_the_answer_it_was_given_as_a_default(dialog):
    prefilled = dialog(BioInputDialog, prompt="name", default="BRCA2")
    assert prefilled._entry.get() == "BRCA2"

    blank = dialog(BioInputDialog, prompt="name")

    assert blank._entry.get() == ""


def test_ok_returns_the_typed_answer_without_the_spaces_around_it(dialog):
    d = dialog(BioInputDialog, prompt="name", default="  BRCA2  ")

    _button(d, "OK").invoke()

    assert d.result == "BRCA2"
    assert not d.winfo_exists()


def test_an_edited_default_is_the_answer_that_comes_back(dialog):
    d = dialog(BioInputDialog, prompt="name", default="BRCA2")
    d._entry.delete(0, "end")
    d._entry.insert(0, "TP53")

    _button(d, "OK").invoke()

    assert d.result == "TP53"


def test_a_blanked_input_answers_an_empty_string_rather_than_no_answer(dialog):
    typed = dialog(BioInputDialog, prompt="name", default="BRCA2")
    _button(typed, "OK").invoke()
    assert typed.result == "BRCA2"

    cleared = dialog(BioInputDialog, prompt="name", default="BRCA2")
    cleared._entry.delete(0, "end")
    _button(cleared, "OK").invoke()
    assert cleared.result == ""

    cancelled = dialog(BioInputDialog, prompt="name", default="BRCA2")
    cancelled._entry.delete(0, "end")
    _button(cancelled, "Cancel").invoke()

    assert cancelled.result is None


def test_the_cancel_button_gives_no_answer_to_the_caller(dialog):
    d = dialog(BioInputDialog, prompt="name", default="BRCA2")

    _button(d, "Cancel").invoke()

    assert d.result is None
    assert not d.winfo_exists()


def test_browse_types_the_chosen_path_into_the_entry_and_ok_answers_it(dialog, monkeypatch):
    asked = {}

    def fake_open(**kwargs):
        asked.update(kwargs)
        return "C:/reads/sample_12.fa"

    monkeypatch.setattr(tkinter.filedialog, "askopenfilename", fake_open)
    d = dialog(BioFilePickerDialog, prompt="pick", filetypes=[("FASTA", "*.fa")])

    _button(d, "Browse").invoke()

    assert d._entry.get() == "C:/reads/sample_12.fa"
    assert asked["filetypes"] == [("FASTA", "*.fa")]

    _button(d, "OK").invoke()
    assert d.result == "C:/reads/sample_12.fa"


def test_a_picker_that_asks_for_no_types_offers_everything(dialog, monkeypatch):
    asked = {}

    def fake_open(**kwargs):
        asked.update(kwargs)
        return ""

    monkeypatch.setattr(tkinter.filedialog, "askopenfilename", fake_open)
    d = dialog(BioFilePickerDialog, prompt="pick")

    _button(d, "Browse").invoke()

    assert asked["filetypes"] == [("All", "*.*")]


def test_a_cancelled_browse_keeps_the_path_the_user_typed(dialog, monkeypatch):
    d = dialog(BioFilePickerDialog, prompt="pick")
    d._entry.insert(0, "C:/keep/me.fa")
    monkeypatch.setattr(tkinter.filedialog, "askopenfilename", lambda **kwargs: "")

    _button(d, "Browse").invoke()

    assert d._entry.get() == "C:/keep/me.fa"

    monkeypatch.setattr(tkinter.filedialog, "askopenfilename", lambda **kwargs: "C:/other.fa")
    _button(d, "Browse").invoke()
    assert d._entry.get() == "C:/other.fa", "browse writes when a file really is chosen"


def test_an_empty_file_entry_answers_no_path_at_all(dialog):
    blank = dialog(BioFilePickerDialog, prompt="pick")
    blank._entry.insert(0, "   ")
    _button(blank, "OK").invoke()
    assert blank.result is None

    typed = dialog(BioFilePickerDialog, prompt="pick")
    typed._entry.insert(0, "C:/x.fa")
    _button(typed, "OK").invoke()

    assert typed.result == "C:/x.fa"


def test_cancelling_the_file_picker_answers_no_path(dialog):
    d = dialog(BioFilePickerDialog, prompt="pick")
    d._entry.insert(0, "C:/x.fa")

    _button(d, "Cancel").invoke()

    assert d.result is None
    assert not d.winfo_exists()


def test_the_dropdown_starts_on_the_default_when_one_is_offered(dialog):
    with_default = dialog(BioDropdownDialog, prompt="pick", options=["a", "b", "c"], default="c")
    assert with_default._combo.get() == "c"

    no_default = dialog(BioDropdownDialog, prompt="pick", options=["a", "b", "c"])

    assert no_default._combo.get() == "a"


def test_the_dropdown_only_offers_its_own_options(dialog):
    d = dialog(BioDropdownDialog, prompt="pick", options=["a", "b"])

    assert d._combo.cget("state") == "readonly"


def test_a_default_outside_the_options_is_refused_before_a_window_exists(dialog):
    with pytest.raises(ValueError, match="not among the options"):
        dialog(BioDropdownDialog, prompt="pick", options=["a"], default="stale")


def test_ok_with_no_options_answers_nothing(dialog):
    d = dialog(BioDropdownDialog, prompt="pick", options=[])

    _button(d, "OK").invoke()

    assert d.result is None


def test_ok_returns_whatever_the_dropdown_shows_at_that_moment(dialog):
    untouched = dialog(BioDropdownDialog, prompt="pick", options=["a", "b", "c"])
    _button(untouched, "OK").invoke()
    assert untouched.result == "a"

    changed = dialog(BioDropdownDialog, prompt="pick", options=["a", "b", "c"], default="c")
    changed._combo.set("b")
    _button(changed, "OK").invoke()

    assert changed.result == "b"


def test_cancelling_the_dropdown_answers_no_answer_even_with_no_options(dialog):
    d = dialog(BioDropdownDialog, prompt="pick", options=[])

    _button(d, "Cancel").invoke()

    assert d.result is None
    assert not d.winfo_exists()


def test_a_dialog_keeps_its_answer_when_the_grab_release_is_refused(dialog, monkeypatch):
    """Real Tk answers a refused release with TclError, which the guard does not
    catch, so the refusal is injected: what is under test is the code between the
    click and destroy()."""
    cases = [
        (BioMessageDialog, dict(message=SHORT, msg_type="info"), "OK", "ok"),
        (BioConfirmDialog, dict(message=SHORT), "Yes", True),
        (BioConfirmDialog, dict(message=SHORT), "Cancel", False),
        (BioInputDialog, dict(prompt="p", default="  typed  "), "OK", "typed"),
        (BioInputDialog, dict(prompt="p", default="x"), "Cancel", None),
        (BioFilePickerDialog, dict(prompt="p"), "OK", None),
        (BioFilePickerDialog, dict(prompt="p"), "Cancel", None),
        (BioDropdownDialog, dict(prompt="p", options=["one", "two"]), "OK", "one"),
        (BioDropdownDialog, dict(prompt="p", options=["one"]), "Cancel", None),
    ]

    for cls, kwargs, label, expected in cases:
        d = dialog(cls, **kwargs)
        monkeypatch.setattr(d, "grab_release", _refuse, raising=False)
        _button(d, label).invoke()
        assert d.result == expected, f"{cls.__name__} {label}"
        assert not d.winfo_exists()


def test_a_dialog_closed_from_the_title_bar_survives_a_refused_grab(dialog, monkeypatch):
    d = dialog(BioMessageDialog, message=SHORT, msg_type="error")
    monkeypatch.setattr(d, "grab_release", _refuse, raising=False)

    d.tk.eval(d.protocol("WM_DELETE_WINDOW"))

    assert d.result is None
    assert not d.winfo_exists()


def test_the_splash_reports_the_step_it_is_asked_to_show(dialog):
    d = dialog(BioSplashScreen)
    before = d._status.cget("text")

    d.update_status("Loading alignment module...", 0.4)

    assert d._status.cget("text") == "Loading alignment module..."
    assert d._progress.get() == pytest.approx(0.4)
    assert d._status.cget("text") != before


def test_the_splash_shows_the_version_it_was_built_with(dialog):
    from biosuite import __version__

    d = dialog(BioSplashScreen)

    assert f"Ultra v{__version__}" in _label_texts(d)


def test_the_splash_fades_one_step_per_tick_and_closes_at_the_floor(dialog):
    d = dialog(BioSplashScreen)
    d.attributes("-alpha", 1.0)

    d.animate_out()

    assert d.attributes("-alpha") == pytest.approx(0.9)
    assert d.winfo_exists()

    d.attributes("-alpha", 0.05)
    d._fade_out()

    assert not d.winfo_exists()
