"""Sharing a plot: the Plotly helpers on the main window.

``_show_interactive_plot`` writes the figure to a temporary HTML file and asks the
browser to open it, ``_save_interactive_plot`` lets the user choose the file, and
``_gui_interactive_plot_api`` renders a demo volcano through ``plot_api`` and hands
it to the first.  Two things are enforced here: the browser is never actually
launched during a test, and nothing is ever written outside ``tmp_path``.
"""
import os

import numpy as np
import pytest

from biosuite.plotting import plot_api


class _Figure:
    """A Plotly-shaped stand-in that records where it was asked to write."""

    def __init__(self, raises=None):
        self.written = []
        self.raises = raises

    def write_html(self, path, **kwargs):
        if self.raises:
            raise self.raises
        self.written.append((path, kwargs))


@pytest.fixture
def browser(monkeypatch, tmp_path):
    """Redirect the temp directory and refuse to open anything."""
    import tempfile
    import webbrowser

    opened = []
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))
    return opened


def test_a_plotly_figure_is_written_and_handed_to_the_browser(ui, browser, tmp_path):
    fig = _Figure()

    ui.app._show_interactive_plot(fig, "Volcano Plot")

    assert len(fig.written) == 1
    path, kwargs = fig.written[0]
    assert path.startswith(str(tmp_path))
    assert path.endswith(".html")
    assert kwargs == {"auto_open": False}
    assert browser == [f"file://{path}"]
    assert ui.status() == f"Interactive plot opened in browser: Volcano Plot"


def test_without_plotly_the_user_is_told_how_to_get_it(ui, browser, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "plotly.io", None)

    ui.app._show_interactive_plot(_Figure(), "Volcano Plot")

    assert ui.messages_of("warning") == ["Install plotly for interactive plots: pip install plotly"]
    assert browser == []


def test_a_figure_that_cannot_be_written_reports_the_reason(ui, browser):
    fig = _Figure(raises=OSError("disk is write-protected"))

    ui.app._show_interactive_plot(fig, "PCA Plot")

    assert ui.messages_of("error") ==             [f"Failed to display interactive plot: disk is write-protected"]
    assert browser == []


def test_saving_uses_the_path_the_dialog_returned(ui, tmp_path):
    target = tmp_path / "my_plot.html"
    fig = _Figure()
    ui.save_path = str(target)

    ui.app._save_interactive_plot(fig, "my_plot")

    assert fig.written == [(str(target), {})]
    assert ui.status() == f"Saved: {target}"
    assert ui.messages_of("success") == [f"Interactive plot saved to:\n{target}"]


def test_a_cancelled_save_dialog_writes_nothing(ui, tmp_path):
    fig = _Figure()
    ui.save_path = ""

    ui.app._save_interactive_plot(fig, "plot")

    assert fig.written == []
    assert ui.messages == []
    assert list(tmp_path.iterdir()) == []


def test_a_failed_save_names_the_reason(ui, tmp_path):
    fig = _Figure(raises=RuntimeError("permission denied"))
    ui.save_path = str(tmp_path / "nope.html")

    ui.app._save_interactive_plot(fig, "nope")

    assert ui.messages_of("error") == ["Failed to save: permission denied"]


def test_the_demo_volcano_reaches_the_browser(ui, browser, tmp_path):
    """The whole path is real: plot_api draws a Plotly figure, the helper writes it
    under tmp_path, and the file the browser was pointed at exists on disk."""
    ui.app._gui_interactive_plot_api()

    assert len(browser) == 1
    written = browser[0].replace("file://", "")
    assert written.startswith(str(tmp_path)), written
    assert os.path.exists(written), "the announced HTML file was never written"
    assert os.path.getsize(written) > 1000, "an empty document was written"
    assert ui.status() == "Interactive plot opened in browser: Volcano Plot"
    assert ui.messages == []


def test_a_demo_that_cannot_be_rendered_is_reported(ui, browser, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("plotly refused the figure")

    monkeypatch.setattr(plot_api, "volcano", broken)

    ui.app._gui_interactive_plot_api()

    assert ui.messages_of("error") == ["plotly refused the figure"]
    assert browser == []


def test_the_demo_volcano_leaves_the_global_rng_alone(ui, browser):
    """The demo reseeded the process RNG with ``np.random.seed(42)``; the demo
    arrays must come from a local generator."""
    before = np.random.get_state()

    ui.app._gui_interactive_plot_api()

    after = np.random.get_state()
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]
    assert len(browser) == 1
