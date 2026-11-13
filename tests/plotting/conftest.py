"""Fixtures for the interactive plotting demos.

The demo entry points read answers from ``input()`` and save through
``utils.ask_save_plot``, and which questions they ask depends on the answers to
the earlier ones, so a flat list of answers breaks whenever a branch changes.
The responder here keys answers on the prompt text and rejects any prompt a test
did not declare, which pins the interactive contract.
"""
import builtins
import contextlib
import io

import matplotlib.pyplot as plt
import pytest

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf")

#: Asked by every demo that reaches the save step.
SAVE_PROMPTS = {"save this plot": "y", "filename (default": ""}

#: Key for a bare ``input()`` call, used by free-form line loops (motif sequences).
BLANK = "<blank>"

#: Table value that makes the prompt raise ``EOFError``, i.e. a closed stdin.
EOF = "<<eof>>"


def responder(answers):
    """Return an ``input`` replacement that answers from a prompt→answer table.

    A table value may be a list, consumed one answer per matching prompt, which
    is how a loop that asks the same question repeatedly is scripted.
    """
    table = {key.lower(): value for key, value in answers.items()}
    queues = {key: list(value) for key, value in table.items()
              if isinstance(value, (list, tuple))}

    def _answer(value):
        if value is EOF:
            raise EOFError
        return str(value)

    def ask(prompt=""):
        if prompt.strip():
            key = next((k for k in table if k != BLANK and k in prompt.lower()), None)
        else:
            key = BLANK if BLANK in table else None
        if key is None:
            raise AssertionError(f"undeclared prompt: {prompt!r}")
        if key in queues:
            if not queues[key]:
                raise AssertionError(f"no answers left for: {prompt!r}")
            return _answer(queues[key].pop(0))
        return _answer(table[key])

    return ask


def figure_facts(fig):
    """What a saved figure actually shows, read while it is still current.

    The demos close their figure inside ``ask_save_plot``, so a test cannot look at
    it afterwards; this is captured from the ``savefig`` spy instead.
    """
    ax = fig.axes[0] if fig.axes else None
    if ax is None:
        return {"figure_title": None, "legend_title": None, "points_drawn": 0,
                "x_label": None, "y_label": None}
    legend = ax.get_legend()
    legend_title = legend.get_title().get_text() if legend is not None else ""
    return {
        "figure_title": ax.get_title() or None,
        "legend_title": legend_title or None,
        "points_drawn": sum(len(collection.get_offsets())
                            for collection in ax.collections),
        "x_label": ax.get_xlabel() or None,
        "y_label": ax.get_ylabel() or None,
    }


@pytest.fixture(autouse=True)
def _isolated_session():
    """Hide the user session from the demos.

    ``safe_float_input`` and the file prompts read defaults out of the shared
    ``session`` dict, so without this a test would inherit whatever an earlier
    demo stored there.
    """
    from biosuite.core.utils import session

    saved = dict(session)
    session.clear()
    yield
    session.clear()
    session.update(saved)


@pytest.fixture
def figure_at_save(monkeypatch):
    """Read a demo's figure at the moment it is about to be saved.

    ``ask_save_plot`` closes the figure it saves (F2-3), so a test cannot look at
    it after the demo returns.  Spying the saver hands it over while it is current.
    """
    captured = {}

    def watch(module):
        real = module.ask_save_plot

        def spy(default_name, save_format, dpi, pdf=None, story=None):
            captured["name"] = default_name
            captured["fig"] = plt.gcf()
            return real(default_name, save_format, dpi, pdf, story)

        monkeypatch.setattr(module, "ask_save_plot", spy)
        return captured

    return watch


@pytest.fixture
def eof():
    """The scripted answer that makes the next prompt raise ``EOFError``."""
    return EOF


@pytest.fixture
def run_demo(tmp_path, monkeypatch):
    """Run one demo in an empty directory and report the artefacts it left.

    The demos catch their own exceptions, so a caller that only checks for a
    raised error passes on a broken plot; ``stdout`` is captured so the internal
    ``Error:`` line is visible to the test.
    """
    import biosuite.core.utils as core_utils

    def _run(func, *args, answers=None, quiet=True, save=True):
        table = dict(SAVE_PROMPTS if save else {"save this plot": "n"})
        table.update(answers or {})
        monkeypatch.setattr(builtins, "input", responder(table))
        monkeypatch.setitem(core_utils.config, "quiet", quiet)
        monkeypatch.chdir(tmp_path)
        plt.close("all")
        facts = {}
        real_savefig = plt.savefig

        def spy_savefig(*call_args, **call_kwargs):
            # The figure is still current here; after ask_save_plot it is closed.
            if not facts:
                facts.update(figure_facts(plt.gcf()))
            return real_savefig(*call_args, **call_kwargs)

        monkeypatch.setattr(plt, "savefig", spy_savefig)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                result = func(*args)
            images = [p for p in tmp_path.iterdir()
                      if p.suffix.lower() in IMAGE_SUFFIXES]
            return {
                "result": result,
                "stdout": stdout.getvalue(),
                "figures_open": len(plt.get_fignums()),
                "images": sorted(p.name for p in images),
                "empty_images": sorted(p.name for p in images if p.stat().st_size == 0),
                **facts,
            }
        finally:
            plt.close("all")

    return _run
