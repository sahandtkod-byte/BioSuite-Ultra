"""A demo that fails halfway must not keep its figure.

Every demo ended in ``except Exception as e: print(f"Error: {e}")``.  That line is
right about the CLI (a menu session has to survive a broken plot) and eighteen
copies of it were still wrong in three ways: the traceback was thrown away with no
logger in the module to keep it, the message named no demo, and a figure created
before the failure was never closed — measured with ``scatter`` on a column of
identical x values and with ``circos`` on zero-sized sectors, both of which left a
figure pinned open.  In a long CLI session that is a process-lifetime leak.
"""
import matplotlib
matplotlib.use('Agg')

import builtins
import io
import logging
import contextlib

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _raise_at_save(monkeypatch, message):
    def spy(default_name, save_format, dpi, pdf=None, story=None):
        raise RuntimeError(message)

    monkeypatch.setattr(bp, 'ask_save_plot', spy)
    answers = {"load data from file": 'n', "use default": 'y', "number of sets": '2',
               "save this plot": 'y', "filename (default": ''}

    def ask(prompt=""):
        for key, value in answers.items():
            if key in prompt.lower():
                return value
        return ''

    monkeypatch.setattr(builtins, 'input', ask)
    monkeypatch.setitem(bp.config, 'quiet', False)


def test_a_failure_after_the_figure_is_drawn_releases_it(monkeypatch, capsys):
    _raise_at_save(monkeypatch, "disk full")
    before = set(plt.get_fignums())

    with contextlib.redirect_stdout(io.StringIO()):
        bp.volcano_plot()

    out = capsys.readouterr().out
    assert out == "", "the demo printed nothing, so where did the line go?" 
    assert set(plt.get_fignums()) == before, \
        f"the demo left {sorted(set(plt.get_fignums()) - before)} open"


def test_the_failing_demo_is_named_in_the_log(monkeypatch):
    _raise_at_save(monkeypatch, "boom")
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    # The guard is shared with specialized_plots since cluster 30, so the record
    # comes from the guard's own logger; the demo's name is in the message.
    logger = logging.getLogger('biosuite.plotting._demo_guard')
    handler = Catch(level=logging.ERROR)
    logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bp.pca_plot()
    finally:
        logger.removeHandler(handler)

    errors = [r for r in records if r.levelno == logging.ERROR]
    assert len(errors) == 1, [r.getMessage() for r in records]
    assert 'pca_plot failed' in errors[0].getMessage(), errors[0].getMessage()
    assert errors[0].name == 'biosuite.plotting._demo_guard'
    assert errors[0].exc_info is not None and errors[0].exc_info[0] is RuntimeError, \
        "the traceback has to survive into the log"


def test_the_user_visible_line_is_unchanged(monkeypatch, capsys):
    """The closed-stdin contract pins the exact text; a re-format would break it."""
    _raise_at_save(monkeypatch, "EOF when reading a line")

    with contextlib.redirect_stdout(io.StringIO()) as out:
        bp.manhattan_plot()

    text = out.getvalue()
    assert "Error: EOF when reading a line" in text, repr(text)
    assert "Traceback" not in text, repr(text)


def test_a_demo_that_succeeds_leaves_nothing_open(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    before = set(plt.get_fignums())
    answers = {"load data from file": 'n', "save this plot": 'y',
               "filename (default": ''}

    def ask(prompt=""):
        for key, value in answers.items():
            if key in prompt.lower():
                return value
        return ''

    monkeypatch.setattr(builtins, 'input', ask)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        bp.ma_plot()

    out = printed.getvalue()
    assert "Error:" not in out, out
    assert set(plt.get_fignums()) == before, "a saved demo must close its figure"
    assert [p.name for p in tmp_path.iterdir()] == ["maplot.png"], out

