"""The plot viewer window: embedding a live figure, styling its toolbar, falling back.

``_show_plot_from_figure`` is what turns a rendered figure into a window the user
can zoom, pan and save from.  It opens a Toplevel, so every window built here is
created withdrawn and destroyed at teardown — nothing flashes on screen.  The
embedding is proved on the figure itself (``fig.canvas`` becomes a Tk canvas),
and closing is driven through the window's own ``WM_DELETE_WINDOW`` command.
"""
import tkinter as tk

import customtkinter as ctk
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def hidden_windows(ui, monkeypatch):
    """The real plot window path, with every window created withdrawn.

    ``ui`` replaces ``_show_plot_from_figure`` with a recorder so most tests never
    open a window; this file is about that method, so the patch is lifted here.
    """
    monkeypatch.delattr(ui.app, "_show_plot_from_figure", raising=False)
    monkeypatch.setattr(ui.app, "_plot_history", [])
    monkeypatch.setattr(ui.app, "_plot_history_index", -1)

    made = []

    class Hidden(ctk.CTkToplevel):
        #: CTk defers applying a geometry until the window is mapped, so the
        #: requested sizes are recorded here instead of read back with geometry().
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.withdraw()
            self.requested = []
            made.append(self)

        def geometry(self, value=None):
            if value:
                self.requested.append(value)
            return super().geometry(value)

    monkeypatch.setattr(ctk, "CTkToplevel", Hidden)
    yield made
    for window in made:
        try:
            window.destroy()
        except tk.TclError:
            pass


def _fig(title="Volcano Plot"):
    fig = plt.figure()
    fig.add_subplot().set_title(title)
    return fig


def _close(window):
    """Run the window's own close handler, the way the title-bar X does."""
    command = window.protocol("WM_DELETE_WINDOW")
    assert command, "the window registered no close handler"
    window.tk.eval(command)


def test_the_plot_window_embeds_the_figure_with_its_toolbar(ui, hidden_windows, monkeypatch):
    styled = []
    monkeypatch.setattr(ui.app, "_style_mpl_toolbar",
                        lambda toolbar, T: styled.append(toolbar))
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Volcano Plot")

    assert window.title() == "Volcano Plot"
    assert type(fig.canvas).__name__ == "FigureCanvasTkAgg",             "the figure was not embedded in a Tk canvas"
    assert styled, "the navigation toolbar was never themed"
    assert ui.app._plot_history[-1]['fig'] is fig
    assert ui.app._plot_history[-1]['name'] == "Volcano Plot"
    assert hidden_windows[-1] is window


def test_closing_the_plot_window_releases_figure_and_window(ui, hidden_windows):
    fig = _fig()
    number = fig.number

    window = ui.app._show_plot_from_figure(fig, "PCA Plot")
    _close(window)

    assert not plt.fignum_exists(number), "the figure stayed open after its window closed"
    assert not window.winfo_exists(), "the window was left behind"


def test_the_plot_window_carries_the_toolbar_hint(ui, hidden_windows):
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Heatmap")

    hints = [child.cget("text") for child in window.winfo_children()
             if isinstance(child, ctk.CTkLabel)]
    assert any("Toolbar" in str(text) for text in hints), hints


def test_the_toolbar_buttons_take_the_theme_colours(ui):
    frame = tk.Frame(ui.app)
    button = tk.Button(frame, text="Zoom")
    label = tk.Label(frame, text="info")
    inner = tk.Frame(frame)
    for child in (button, label, inner):
        child.pack()
    T = ui.app.T

    ui.app._style_mpl_toolbar(frame, T)

    assert frame.cget("background") == T['card']
    assert button.cget("background") == "#dfe3ea", "icons ship dark, so the chips must stay light"
    assert button.cget("activebackground") == "#c3c9d4"
    assert label.cget("background") == T['card']
    assert label.cget("foreground") == T['text']
    assert inner.cget("background") == T['card']
    frame.destroy()


def test_the_message_label_of_a_real_toolbar_is_accent_coloured(ui):
    frame = tk.Frame(ui.app)
    message = tk.Label(frame)
    frame._message_label = message
    T = ui.app.T

    ui.app._style_mpl_toolbar(frame, T)

    assert message.cget("foreground") == T['accent']
    frame.destroy()


def test_a_toolbar_that_refuses_to_be_styled_does_not_cost_the_user_the_plot(
        ui, hidden_windows, monkeypatch):
    def refuse(*args, **kwargs):
        raise tk.TclError("unknown color name")

    import matplotlib.backends.backend_tkagg as tkagg

    original = tkagg.FigureCanvasTkAgg

    class _Canvas(original):
        def get_tk_widget(self):
            widget = super().get_tk_widget()
            widget.configure = refuse
            return widget

    monkeypatch.setattr(tkagg, "FigureCanvasTkAgg", _Canvas)
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Barplot")

    assert window.title() == "Barplot"
    assert isinstance(fig.canvas, original), "the themed canvas is no longer a Tk canvas"


def test_a_toolbar_that_cannot_be_read_is_left_alone(ui):
    """The styling pass only touches a toolbar it can read; anything else must not
    stop the plot from opening."""
    attempted = []

    class _Broken:
        def configure(self, *args, **kwargs):
            attempted.append((args, kwargs))
            raise tk.TclError("widget is gone")

    ui.app._style_mpl_toolbar(_Broken(), ui.app.T)

    assert attempted, \
        "the toolbar was never offered any styling, so swallowing nothing proved nothing"


def test_closing_a_window_whose_figure_will_not_close_still_destroys_it(
        ui, hidden_windows, monkeypatch):
    fig = _fig()
    window = ui.app._show_plot_from_figure(fig, "UMAP Plot")
    real_close = plt.close

    def refuse(target=None):
        if target is fig:
            raise RuntimeError("the figure manager is gone")
        return real_close(target)

    monkeypatch.setattr(plt, "close", refuse)

    _close(window)

    assert not window.winfo_exists(), "the window stayed open behind the failing close"


def test_a_figure_that_cannot_be_embedded_is_shown_as_a_picture(
        ui, hidden_windows, monkeypatch, tmp_path):
    import tempfile

    import matplotlib.backends.backend_tkagg as tkagg

    def refuse(*args, **kwargs):
        raise RuntimeError("no display for this thread")

    monkeypatch.setattr(tkagg, "FigureCanvasTkAgg", refuse)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Circos Plot")

    assert (tmp_path / f"biosuite_{id(fig)}.png").exists(),             "the fallback never rendered the figure"
    attached = [child for child in window.winfo_children()
                if getattr(child, "_photo_ref", None)]
    assert attached, "the static image was never attached to the window"
    assert attached[0]._photo_ref[0] is not None


def test_a_broken_fallback_says_so_instead_of_showing_an_empty_window(
        ui, hidden_windows, monkeypatch, tmp_path):
    import tempfile

    import matplotlib.backends.backend_tkagg as tkagg
    from PIL import Image

    monkeypatch.setattr(tkagg, "FigureCanvasTkAgg",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    def refuse(*args, **kwargs):
        raise OSError("Pillow cannot read this image")

    monkeypatch.setattr(Image, "open", refuse)
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Sankey")

    texts = [str(child.cget("text")) for child in window.winfo_children()
             if isinstance(child, ctk.CTkLabel)]
    assert any("Pillow cannot read this image" in text for text in texts), texts


def test_the_plot_window_is_still_sized_when_the_screen_geometry_is_unavailable(
        ui, hidden_windows, monkeypatch):
    """Centring needs the main window's position; when that cannot be read the
    viewer must still get its own size."""
    def refuse(*args, **kwargs):
        raise tk.TclError("cannot query a withdrawn window")

    monkeypatch.setattr(ui.app, "winfo_rootx", refuse)
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Ridge Plot")

    assert window.requested[-1] == "980x780",             "the viewer fell back to an uncentred size"


def test_the_plot_window_is_centred_over_the_main_window(ui, hidden_windows):
    fig = _fig()

    window = ui.app._show_plot_from_figure(fig, "Dot Plot")

    assert window.requested[-1].startswith("980x780+"),             "no offset means the centring was skipped"
    _, size = window.requested[-1].split("+", 1)
    assert size.count("+") == 1, window.requested[-1]
