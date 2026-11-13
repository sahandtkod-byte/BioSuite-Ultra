"""The app's dialog wrappers: do they build the themed dialog they claim to?

``ui`` normally disarms ``_msg_*``, ``_confirm`` and the ``_ask_*`` helpers so no
test can block.  These tests remove the disarming, replace the dialog *classes*
with recorders, and call the real wrapper bodies — which is the only way to check
that the warning wrapper asks for a warning dialog, that ``_confirm`` only answers
True to a True, and that the prompt, default, options and file types reach the
widget that shows them.
"""
import pytest

from biosuite.gui import main_window


@pytest.fixture
def real_wrappers(ui, monkeypatch):
    """The class-level wrappers, with the dialogs replaced by recorders."""
    for kind in ("info", "warning", "error", "success"):
        monkeypatch.delattr(ui.app, f"_msg_{kind}", raising=False)
    for name in ("_confirm", "_ask_input", "_ask_dropdown", "_ask_file"):
        monkeypatch.delattr(ui.app, name, raising=False)
    monkeypatch.setattr(ui.app, "wait_window", lambda dialog: None)
    return ui.app


@pytest.fixture
def recorded(monkeypatch):
    """Every dialog class swapped for a recorder; the list fills as they are built."""
    made = []
    monkeypatch.setattr(main_window, "BioMessageDialog",
                        lambda *args, **kwargs: made.append(("message", args, kwargs)))
    monkeypatch.setattr(main_window, "BioConfirmDialog",
                        lambda *args, **kwargs: made.append(("confirm", args, kwargs)))
    monkeypatch.setattr(main_window, "BioInputDialog",
                        lambda *args, **kwargs: made.append(("input", args, kwargs)))
    monkeypatch.setattr(main_window, "BioDropdownDialog",
                        lambda *args, **kwargs: made.append(("dropdown", args, kwargs)))
    monkeypatch.setattr(main_window, "BioFilePickerDialog",
                        lambda *args, **kwargs: made.append(("file", args, kwargs)))
    return made


class _Answered:
    """A dialog that hands back a fixed result and remembers nothing else."""

    def __init__(self, result=None):
        self.result = result

    def destroy(self):
        pass


def test_each_message_kind_builds_a_dialog_of_that_kind(real_wrappers, recorded):
    for kind in ("info", "warning", "error", "success"):
        getattr(real_wrappers, f"_msg_{kind}")("Title", "Body text")

    built = [(entry[2]['title'], entry[2]['message'], entry[2]['msg_type'])
             for entry in recorded]
    assert built == [("Title", "Body text", "info"),
                     ("Title", "Body text", "warning"),
                     ("Title", "Body text", "error"),
                     ("Title", "Body text", "success")]
    assert all(entry[1][0] is real_wrappers for entry in recorded),             "the dialog is not parented to the app"
    assert all(entry[1][1] is real_wrappers.T for entry in recorded),             "the dialog lost the active theme"


def test_a_yes_answer_is_the_only_way_confirm_returns_true(real_wrappers, monkeypatch):
    monkeypatch.setattr(main_window, "BioConfirmDialog", lambda *a, **k: _Answered(True))
    assert real_wrappers._confirm("Q", "Delete the file?") is True
    monkeypatch.setattr(main_window, "BioConfirmDialog", lambda *a, **k: _Answered(False))
    assert real_wrappers._confirm("Q", "Delete the file?") is False
    monkeypatch.setattr(main_window, "BioConfirmDialog", lambda *a, **k: _Answered("yes"))
    assert real_wrappers._confirm("Q", "Delete the file?") is False,             "a non-boolean answer must not count as agreement"


def test_the_input_dialog_gets_the_prompt_and_the_default(real_wrappers, monkeypatch):
    seen = {}

    def factory(parent, T, title="", prompt="", default=""):
        seen.update(title=title, prompt=prompt, default=default)
        return _Answered("GATC")

    monkeypatch.setattr(main_window, "BioInputDialog", factory)

    assert real_wrappers._ask_input("Primers", "Enter a sequence:", default="AAAA") == "GATC"
    assert seen == {"title": "Primers", "prompt": "Enter a sequence:", "default": "AAAA"}


def test_the_dropdown_dialog_gets_the_offered_options(real_wrappers, monkeypatch):
    seen = {}

    def factory(parent, T, title="", prompt="", options=None, default=None):
        seen.update(options=options, default=default)
        return _Answered("bed")

    monkeypatch.setattr(main_window, "BioDropdownDialog", factory)

    assert real_wrappers._ask_dropdown("Track", "Which track?", ["bed", "vcf"], "vcf") == "bed"
    assert seen == {"options": ["bed", "vcf"], "default": "vcf"}


def test_the_file_dialog_gets_the_file_types_it_must_filter_by(real_wrappers, monkeypatch):
    seen = {}

    def factory(parent, T, title="", prompt="", filetypes=None):
        seen.update(title=title, prompt=prompt, filetypes=filetypes)
        return _Answered(r"X:\reads.fq")

    monkeypatch.setattr(main_window, "BioFilePickerDialog", factory)

    picked = real_wrappers._ask_file("Open", "Pick a FASTQ", [("FASTQ", "*.fq")])
    assert picked == r"X:\reads.fq"
    assert seen == {"title": "Open", "prompt": "Pick a FASTQ",
                    "filetypes": [("FASTQ", "*.fq")]}


def test_a_dismissed_dialog_hands_back_its_own_none(real_wrappers, monkeypatch):
    monkeypatch.setattr(main_window, "BioInputDialog", lambda *a, **k: _Answered(None))
    monkeypatch.setattr(main_window, "BioDropdownDialog",
                        lambda *a, **k: _Answered(None))
    monkeypatch.setattr(main_window, "BioFilePickerDialog",
                        lambda *a, **k: _Answered(None))

    assert real_wrappers._ask_input("T", "p") is None
    assert real_wrappers._ask_dropdown("T", "p", ["a"]) is None
    assert real_wrappers._ask_file("T", "p") is None
