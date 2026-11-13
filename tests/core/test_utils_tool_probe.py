"""`has_tool` — the question "can this program be started?", and nothing else.

Three answers were wrong before: a tool that was on `PATH` but exited non-zero for
`--version` (several bioinformatics programs print usage and exit ``1``/``2``) was
reported as **not installed**, so the package fell back to a slower built-in engine
beside a working binary; a tool that could not be started because of a permission
problem raised `PermissionError` out of every module that asks this question; and
the probe itself cost a subprocess per call even for a name that is nowhere on
`PATH`.  Presence now comes from `shutil.which`, and the probe only decides whether
the program actually ran.
"""
import subprocess

import pytest

from biosuite.core import utils


class _Run:
    def __init__(self, returncode=0, raises=None):
        self.returncode = returncode
        self.raises = raises
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append((list(cmd), kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, "", "")


@pytest.fixture
def patched(monkeypatch):
    """`shutil.which` says 'found', `subprocess.run` is a recorder."""
    def apply(which_result="C:/tools/bwa", run=None):
        monkeypatch.setattr(utils.shutil, "which", lambda name: which_result)
        recorder = run if run is not None else _Run()
        monkeypatch.setattr(utils.subprocess, "run", recorder)
        return recorder
    return apply


def test_a_name_that_is_nowhere_on_path_never_reaches_the_probe(patched):
    recorder = patched(which_result=None)
    assert utils.has_tool("absent_tool_xyz") is False
    assert recorder.commands == [], "no program, no subprocess"


def test_a_tool_that_exits_non_zero_for_version_is_still_installed(patched):
    """Measured on the old code: `return r.returncode == 0` turned a usage-exiting
    `bwa --version` into "not available"."""
    recorder = patched(run=_Run(returncode=2))
    assert utils.has_tool("bwa") is True
    assert recorder.commands[0][0] == ["bwa", "--version"]


def test_a_tool_that_cannot_be_started_is_reported_absent(patched):
    """A deleted or non-executable file used to raise `PermissionError`/`OSError`
    straight through every caller."""
    for error in (PermissionError(13, "Permission denied"),
                  FileNotFoundError(2, "No such file or directory"),
                  OSError(8, "Exec format error")):
        patched(run=_Run(raises=error))
        assert utils.has_tool("bwa") is False, error


def test_a_probe_that_hangs_is_answered_as_absent(patched):
    patched(run=_Run(raises=subprocess.TimeoutExpired(cmd="bwa", timeout=10)))
    assert utils.has_tool("bwa") is False


def test_the_probe_waits_no_longer_than_ten_seconds(patched):
    recorder = patched()
    assert utils.has_tool("bwa") is True
    assert recorder.commands[0][1]["timeout"] == 10


@pytest.mark.parametrize("name, message", [
    pytest.param(None, "has_tool needs a command name, got None.", id="none"),
    pytest.param("", "has_tool needs a command name, got ''.", id="empty"),
    pytest.param("   ", "has_tool needs a command name, got '   '.", id="blank"),
    pytest.param(42, "has_tool needs a command name, got 42.", id="number"),
    pytest.param(["bwa"], "has_tool needs a command name, got ['bwa'].", id="list"),
])
def test_a_name_that_is_not_a_command_is_refused(name, message):
    with pytest.raises(ValueError) as excinfo:
        utils.has_tool(name)
    assert str(excinfo.value) == message


def test_a_program_that_is_really_here_answers_true():
    assert utils.has_tool("python") is True


def test_a_program_that_is_really_not_here_answers_false():
    assert utils.has_tool("biosuite-no-such-tool-xyz-123") is False
