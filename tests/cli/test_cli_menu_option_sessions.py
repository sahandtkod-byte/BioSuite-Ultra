"""Every menu option has to survive being chosen with no data and default answers.

``_handle_menu_choice`` is the body of all 116 menu options, and before this file
those branches had never been executed as a group.  An option is allowed to refuse an
empty session — it must then either say so itself or raise, because :func:`main_cli`
turns a raise into a one-line diagnostic and returns to the menu (BSU-014).  What is
not allowed is silence: ten branches (20, 28, 32, 33, 34, 47, h, i, j, p) read an
empty path, got nothing back and returned without a word, which is how this sweep
found them.  ``NameError``, ``UnboundLocalError`` and their siblings are asserted
absent because they are the signature of broken code rather than of an empty session
— options 80, 81 and 96 died that way, from a local ``import matplotlib.pyplot`` in
one branch shadowing the module-level ``plt`` for the whole function.
"""
import builtins
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from biosuite.cli import menu

MENU_SOURCE = Path(menu.__file__).read_text(encoding="utf-8")

#: Every key the dispatch chain of ``_handle_menu_choice`` handles, read from the
#: source so a new option is picked up automatically instead of being forgotten
#: here.  The slice stops the sweep from adopting the loop's own ``0`` (exit), which
#: is not an action; ``menu.py`` displays 117 entries and 116 of them are options.
HANDLER_SOURCE = MENU_SOURCE[MENU_SOURCE.index("def _handle_menu_choice"):]
OPTIONS = sorted(set(re.findall(r"(?:el)?if choice == '([a-z0-9]+)'", HANDLER_SOURCE)))

#: Exceptions that mean the option itself is broken, not that the session is empty.
#: An ``UnboundLocalError`` is what a local ``import`` shadowing a module global
#: produces, and it reaches the user as a traceback fragment instead of advice.
INTERNAL_ERRORS = (NameError, AttributeError, TypeError, IndexError, UnboundLocalError,
                   ZeroDivisionError, ImportError)


class _StubApp:
    """Stand-in for ``BioSuiteApp`` so option 89 can never open a Tk window."""

    def __init__(self, log):
        self._log = log

    def mainloop(self):
        self._log.append("gui-mainloop")


def _press_enter(prompts, log):
    """Answer every prompt like a user who keeps pressing Enter.

    Nothing is saved, no file path is ever named, and set-count prompts get a small
    legal number.  ``prompts`` collects the questions asked so the caller can tell an
    announced output path from a silent one; ``log`` records the answers the launcher
    tests care about, so a stubbed launch can be told apart from a real one.
    """
    def answer(prompt=""):
        prompts.append(prompt)
        p = prompt.lower()
        if "save this plot" in p or "show data summary" in p:
            return "n"
        if "(y/n)" in p or "y/n" in p:
            return "n"
        if "number of sets" in p:
            return "2"
        if "use default" in p:
            return "y"
        if "gui" in p:
            log.append("gui-prompt")
            return "n"
        return ""

    return answer


@pytest.fixture
def empty_session(tmp_path, monkeypatch, capsys):
    """Choose one option in a session that has no data and default answers.

    The option is called directly so the captured stdout is only what the option
    itself said — the menu banner and the exit greeting would make every option look
    talkative.  Both launchers are stubbed so no option can open a window or bind a
    port, and the session runs in ``tmp_path`` so nothing can litter the repository.
    """
    import biosuite.gui.main_window as main_window

    log = []
    prompts = []
    monkeypatch.setattr(menu, "_launch_gui", lambda: log.append("launch_gui"))
    monkeypatch.setattr(menu, "_launch_api", lambda port=None: log.append(f"launch_api:{port}"))
    monkeypatch.setattr(main_window, "BioSuiteApp", lambda *a, **k: _StubApp(log))
    monkeypatch.setitem(menu.config, "quiet", True)
    monkeypatch.setattr(builtins, "input", _press_enter(prompts, log))
    monkeypatch.chdir(tmp_path)

    def run(choice):
        figures_before = set(plt.get_fignums())
        broken, refused = None, None
        try:
            menu._handle_menu_choice(choice)
        except INTERNAL_ERRORS as exc:
            broken = exc
        except Exception as exc:
            refused = exc
        out = capsys.readouterr().out
        written = sorted(str(p.relative_to(Path.cwd()))
                         for p in Path.cwd().rglob("*") if p.is_file())
        leaked = set(plt.get_fignums()) - figures_before

        assert broken is None, \
            f"option {choice} raised {type(broken).__name__}: {broken}"
        return {"out": out, "written": written, "leaked": leaked, "asked": list(prompts),
                "refused": refused, "log": log}

    return run


@pytest.mark.parametrize("choice", OPTIONS)
def test_option_speaks_or_raises_for_an_empty_session(empty_session, choice):
    """A menu option may refuse an empty session, but not go quietly."""
    result = empty_session(choice)

    assert result["refused"] is not None or result["out"].strip(), \
        f"option {choice} neither printed anything nor raised anything"
    assert not result["leaked"], \
        f"option {choice} left {len(result['leaked'])} figure(s) open"
    # A demo option is allowed to write the file it named out loud — it is not
    # allowed to write one the user never saw mentioned.
    announced = result["out"] + "\n".join(result["asked"])
    silent = [f for f in result["written"] if f not in announced]
    assert not silent, f"option {choice} wrote {silent} without mentioning them"


@pytest.mark.parametrize(("choice", "expected"),
                         [("89", "launch_gui"), ("90", "launch_api:8000")])
def test_launcher_options_delegate_to_menu_helpers(empty_session, choice, expected):
    """Options 89 and 90 must reach the launchers through ``menu``'s own helpers.

    Option 89 inlined ``BioSuiteApp().mainloop()``, duplicating ``_launch_gui`` and
    leaving the option no seam to be stubbed through: running it in a test opened a
    real Tk window for over a hundred seconds.
    """
    result = empty_session(choice)

    assert expected in result["log"], f"option {choice} did not call {expected}"


def test_sweep_still_covers_every_option_the_menu_shows():
    """Guard the collector itself.

    A regex that quietly stopped matching would parametrise zero cases and read as a
    clean run, so the option list is re-derived from what ``print_menu`` displays:
    every displayed entry except ``0`` (exit, which the loop answers) must be swept.
    """
    displayed = MENU_SOURCE[MENU_SOURCE.index("def print_menu"):
                            MENU_SOURCE.index("def main_cli")]
    labels = set(re.findall(r"\{[A-Z]\}\{[A-Z]\} ?([a-z0-9]{1,3})\{R\}", displayed))

    assert OPTIONS == sorted(labels - {"0"})
    assert len(OPTIONS) > 100, f"sweep collected only {len(OPTIONS)} options"
