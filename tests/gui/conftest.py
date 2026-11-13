"""One real ``BioSuiteApp`` for the whole GUI session, with its dialogs disarmed.

The tab handlers only do anything when they are *called*, and the structural tests
that used to stand in for them (`hasattr`, `inspect.getsource`) left 2,753 of the
GUI's 3,143 statements unexecuted.  Driving them needs a real Tk application, which
raises three problems this module solves:

**The window must not appear.**  ``BioSuiteApp`` withdraws itself while building and
schedules ``_show_main_window`` with ``after(300, …)``; ``deiconify``, ``lift`` and
``focus_force`` are replaced with no-ops so nothing ever flashes on screen.

**Nothing may block.**  Every scheduled callback is run synchronously by replacing
``after``, and the four helpers that end in ``self.wait_window(dialog)``
(``_confirm``, ``_ask_input``, ``_ask_dropdown``, ``_ask_file``) would hang the test
run forever, so they return scripted answers and record what was asked.
``tkinter.filedialog.ask*`` — which the tab handlers call directly — is scripted the
same way.

**It has to be fast enough.**  Building the app (37 frames, hundreds of widgets)
takes about 25 s here, so it happens once per session; each test gets the same
instance, and the ``ui`` fixture is what scripts its answers per test.

**Deferred work stays addressable.**  ``after`` returns a fresh non-zero id so the
code paths that only cancel a pending callback (the gallery's debounced search)
behave as they do in the real app; ``after_cancel`` accepts and forgets those ids,
and ``after_idle`` schedules without running, which is what a resize debounce needs
to be observable.
"""
import tkinter as tk

import customtkinter as ctk
import matplotlib
import pytest

matplotlib.use("Agg")

MISSING = object()
SHIMMED = ("after", "after_cancel", "after_idle", "deiconify", "lift", "focus_force")


_BUILT_WIDGETS = []
_BUILT_STATUS = ""
_BUILT_FRAME = "plots"
_BUILT_PLOT_ITEMS = []


def _alive(widget):
    try:
        return bool(widget.winfo_exists())
    except (tk.TclError, AttributeError):
        return False


def _snapshot_built_state(app):
    """Record the texts, status line and tab the app was built with."""
    global _BUILT_STATUS, _BUILT_FRAME, _BUILT_PLOT_ITEMS
    _BUILT_WIDGETS.clear()
    _BUILT_PLOT_ITEMS = list(app._plot_items)
    for name, widget in vars(app).items():
        if not isinstance(widget, (ctk.CTkTextbox, tk.Text, ctk.CTkEntry,
                                   tk.Entry, ctk.CTkComboBox)):
            continue
        try:
            if isinstance(widget, (ctk.CTkTextbox, tk.Text)):
                text = widget.get("1.0", "end").rstrip("\n")
            else:
                text = widget.get()
        except tk.TclError:
            continue
        _BUILT_WIDGETS.append([name, widget, text])
    _BUILT_STATUS = app.status_bar.cget("text")
    _BUILT_FRAME = getattr(app, "_current_frame", "plots")


def _set_widget_text(widget, text):
    # Teardown patches may still be armed; the next setup retries without them.
    try:
        if isinstance(widget, (ctk.CTkTextbox, tk.Text)):
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
        elif isinstance(widget, ctk.CTkComboBox):
            widget.set(text)
        else:
            widget.delete(0, "end")
            widget.insert(0, text)
    except Exception:
        pass


def restore_built_state(app):
    """Return the shared app to the state the session built it in.

    The autouse ``fresh_app_state`` fixture runs this after every test —
    failed ones included — and ``UI.reset`` exposes it for mid-test use.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    children = list(app.winfo_children())
    while children:
        child = children.pop()
        if isinstance(child, tk.Toplevel):
            try:
                child.destroy()
            except Exception:
                pass
        else:
            children.extend(child.winfo_children())
    for entry in _BUILT_WIDGETS:
        name, widget, text = entry
        current = getattr(app, name, None)
        if _alive(current):
            # Rebuilds replace widgets; retain the latest live replacement.
            entry[1] = current
            _set_widget_text(current, text)
        elif _alive(widget):
            setattr(app, name, widget)  # a test's stand-in object is undone
            _set_widget_text(widget, text)
    app._plot_history = []
    app._plot_history_index = -1
    app._progress_bar = None
    app._search_after_id = None
    app._selected_plot_id = None
    app._inter_data = None
    app._bigwig_path = None
    for name in ("_gwas_data", "_resize_scheduled", "_last_digest_result", "_svcnv_data"):
        app.__dict__.pop(name, None)
    if app._selected_cat != "All" or app._plot_items != _BUILT_PLOT_ITEMS:
        try:
            app._select_category("All")
        except Exception:
            pass
    if getattr(app, "_current_frame", None) != _BUILT_FRAME:
        try:
            app._show_frame(_BUILT_FRAME)
        except Exception:
            pass
    try:
        app.status_bar.configure(text=_BUILT_STATUS)
    except Exception:
        pass


@pytest.fixture(scope="session")
def gui_app():
    """A fully built BioSuiteApp: sidebar, content area and every tab frame."""
    from biosuite.gui.main_window import BioSuiteApp

    ids = iter(range(1, 1 << 30))

    def synchronous_after(self, milliseconds, func=None, *args):
        if func is not None:
            func(*args)
        return next(ids)

    def cancelled(self, after_id):
        return None

    def deferred_idle(self, func=None, *args):
        return next(ids)

    saved = {name: BioSuiteApp.__dict__.get(name, MISSING) for name in SHIMMED}
    BioSuiteApp.after = synchronous_after
    BioSuiteApp.after_cancel = cancelled
    BioSuiteApp.after_idle = deferred_idle
    for name in ("deiconify", "lift", "focus_force"):
        setattr(BioSuiteApp, name, lambda self: None)
    try:
        app = BioSuiteApp()
        assert app.frames, "the app built without any tab frames"
        _snapshot_built_state(app)
        yield app
        app.destroy()
    finally:
        for name, value in saved.items():
            if value is MISSING:
                delattr(BioSuiteApp, name)
            else:
                setattr(BioSuiteApp, name, value)


@pytest.fixture(autouse=True)
def fresh_app_state(gui_app):
    """Every GUI test starts from — and is returned to — the as-built app.

    The restore runs after failed tests too: a test that dies halfway must not
    hand its mess to the next one.
    """
    restore_built_state(gui_app)
    yield gui_app
    restore_built_state(gui_app)


@pytest.fixture
def app_reset(gui_app):
    """The teardown's restore, callable mid-test or after an aborted body."""
    return lambda: restore_built_state(gui_app)


class UI:
    """Test-side handle on the app: what it was asked, and what it answered."""

    def __init__(self, app):
        self.app = app
        self.messages = []        # (kind, title, message) from the _msg_* helpers
        self.asked = []           # (kind, prompt) from the blocking helpers
        self.open_path = ""
        self.save_path = ""
        self.directory = ""
        self.confirm_answer = True
        self.input_answer = ""
        self.dropdown_answer = None
        self.plots = []            # (title, figure) the app asked to display

    def _is_textbox(self, widget_name):
        from customtkinter import CTkTextbox

        return isinstance(getattr(self.app, widget_name), CTkTextbox)

    def set(self, widget_name, value):
        """Type into an entry or a results box, the way the user does."""
        widget = getattr(self.app, widget_name)
        if self._is_textbox(widget_name):
            widget.delete("1.0", "end")
            widget.insert("1.0", value)
        else:
            widget.delete(0, "end")
            widget.insert(0, value)

    def text(self, widget_name):
        """The text a widget shows: entries, multi-line boxes and labels."""
        widget = getattr(self.app, widget_name)
        if self._is_textbox(widget_name):
            return widget.get("1.0", "end")
        if hasattr(widget, "get"):
            return widget.get()
        return widget.cget("text")

    def clear(self, widget_name):
        """Empty a widget, because the app is shared for the whole session."""
        widget = getattr(self.app, widget_name)
        if self._is_textbox(widget_name):
            widget.delete("1.0", "end")
        elif hasattr(widget, "delete"):
            widget.delete(0, "end")

    def reset(self):
        """Mid-test return to the as-built state: the teardown's own restore."""
        restore_built_state(self.app)

    def status(self):
        """The status line, without the padding the label adds around it."""
        return self.text("status_bar").strip()

    def messages_of(self, kind):
        return [message for k, _title, message in self.messages if k == kind]

    def last_question(self):
        return self.asked[-1] if self.asked else None


@pytest.fixture
def ui(gui_app, monkeypatch, tmp_path):
    """Disarm every dialog a handler can open; file pickers answer from tmp_path."""
    handle = UI(gui_app)

    for kind in ("info", "warning", "error", "success"):
        monkeypatch.setattr(
            gui_app, f"_msg_{kind}",
            lambda title, message, k=kind: handle.messages.append((k, title, message)))

    def record(kind, prompt, answer):
        handle.asked.append((kind, prompt))
        return answer

    monkeypatch.setattr(gui_app, "_confirm",
                        lambda title, message: record("confirm", message,
                                                      handle.confirm_answer))
    monkeypatch.setattr(gui_app, "_ask_input",
                        lambda title, prompt, default="": record("input", prompt,
                                                                 handle.input_answer))
    monkeypatch.setattr(gui_app, "_ask_dropdown",
                        lambda title, prompt, options, default=None:
                        record("dropdown", prompt, handle.dropdown_answer))
    monkeypatch.setattr(gui_app, "_ask_file",
                        lambda title, prompt, filetypes=None:
                        record("file", prompt, handle.open_path))

    # Plot windows would open a Toplevel per handler; record them instead.  The
    # tabs' own _show_plot_figure shims call this same method, so patching it here
    # still runs the shim.
    def record_plot(fig, title="Plot"):
        handle.plots.append((title, fig))
        return title

    monkeypatch.setattr(gui_app, "_show_plot_from_figure", record_plot)

    import tkinter.filedialog

    monkeypatch.setattr(tkinter.filedialog, "askopenfilename",
                        lambda *args, **kwargs: handle.open_path)
    monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename",
                        lambda *args, **kwargs: handle.save_path)
    monkeypatch.setattr(tkinter.filedialog, "askdirectory",
                        lambda *args, **kwargs: handle.directory)
    monkeypatch.chdir(tmp_path)
    return handle
