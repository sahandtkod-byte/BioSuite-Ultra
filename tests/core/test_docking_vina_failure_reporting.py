"""What the module says when Vina is installed and still gives nothing.

Three different things used to answer with the same sentence: Vina not being
installed, Vina failing to run, and Vina running but writing no poses.  All three
fell through to "AutoDock Vina not found", which sent users to install a program
they already had while the real error — in the subprocess's stderr — was discarded.
"""
import os
import subprocess
import warnings

import pytest

from biosuite.core import docking as dk


@pytest.fixture
def receptor(tmp_path):
    path = tmp_path / "rec.pdb"
    path.write_text("".join(
        f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    "
        f"{i * 3.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C\n"
        for i in range(10)), encoding="utf-8")
    return str(path)


@pytest.fixture
def ligand(tmp_path):
    path = tmp_path / "lig.pdb"
    path.write_text("".join(
        f"HETATM{i + 1:>5}  C1  LIG B{i + 1:>4}    "
        f"{i * 1.5:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C\n"
        for i in range(4)), encoding="utf-8")
    return str(path)


class VinaRun:
    """A stand-in for subprocess.run that writes *body* to the pose file."""

    def __init__(self, returncode=0, body="", stderr="", stdout="", raises=None):
        self.returncode, self.body = returncode, body
        self.stderr, self.stdout, self.raises = stderr, stdout, raises
        self.commands = []
        self.pose_files = []

    def __call__(self, cmd, **kwargs):
        self.commands.append((list(cmd), kwargs))
        out = cmd[cmd.index('--out') + 1]
        self.pose_files.append(out)
        if self.raises:
            raise self.raises
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(self.body)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout,
                                           self.stderr)


def _vina_dock(monkeypatch, fake):
    monkeypatch.setattr(dk.subprocess, "run", fake)
    return dk._vina_dock("receptor.pdbqt", "ligand.pdbqt", [0.0, 0.0, 0.0],
                         (16.0, 16.0, 16.0), 3)


# ── the reason the wrapper hands back ───────────────────────────────────────

def test_a_non_zero_exit_names_the_code_and_the_first_line_of_the_error(monkeypatch):
    fake = VinaRun(returncode=1, stderr="Commandline: ERROR: Parsing of receptor "
                     "file failed\nTraceback (most recent call last):")

    poses, reason = _vina_dock(monkeypatch, fake)

    assert poses == []
    assert reason == ("vina exited with code 1: Commandline: ERROR: Parsing of "
                      "receptor file failed"), reason


def test_a_non_zero_exit_without_any_output_still_names_the_code(monkeypatch):
    poses, reason = _vina_dock(monkeypatch, VinaRun(returncode=2))

    assert poses == [] and reason == "vina exited with code 2", reason


def test_stdout_is_read_when_the_error_stream_is_empty(monkeypatch):
    poses, reason = _vina_dock(monkeypatch, VinaRun(returncode=11,
                                                   stdout="vina: --num_modes: bad value"))

    assert poses == [] and "bad value" in reason, reason


def test_a_binary_that_cannot_be_started_says_so(monkeypatch):
    poses, reason = _vina_dock(monkeypatch,
                               VinaRun(raises=OSError("No such file or directory: 'vina'")))

    assert poses == []
    assert reason.startswith("the vina process could not be run:"), reason
    assert "No such file or directory" in reason, reason


def test_a_timeout_is_a_failure_reason_not_a_crash(monkeypatch):
    timeout = subprocess.TimeoutExpired(cmd="vina", timeout=3600)

    poses, reason = _vina_dock(monkeypatch, VinaRun(raises=timeout))

    assert poses == [] and reason.startswith("the vina process could not be run:"), reason


def test_a_successful_run_that_wrote_no_poses_says_what_was_missing(monkeypatch):
    poses, reason = _vina_dock(monkeypatch, VinaRun(returncode=0, body="MODEL 1\nENDMDL\n"))

    assert poses == []
    assert reason == "vina wrote no 'REMARK VINA RESULT' pose record", reason


def test_a_successful_run_reports_no_failure(monkeypatch):
    body = ("MODEL 1\nREMARK VINA RESULT:    -8.4      0.000  0.000\n"
            "ATOM      1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00\n"
            "ENDMDL\n")

    poses, reason = _vina_dock(monkeypatch, VinaRun(returncode=0, body=body))

    assert reason == "", reason
    assert poses == [(1, -8.4, [1.0, 2.0, 3.0])], poses


@pytest.mark.parametrize("fake", [
    pytest.param(lambda: VinaRun(returncode=1, stderr="boom"), id="exit-code"),
    pytest.param(lambda: VinaRun(raises=OSError("no binary")), id="cannot-start"),
    pytest.param(lambda: VinaRun(returncode=0, body=""), id="no-poses"),
])
def test_every_failure_is_logged_not_only_returned(monkeypatch, caplog, fake):
    """The caller folds the failure into a fallback; the log is what keeps it findable."""
    with caplog.at_level("WARNING", logger="biosuite.core.docking"):
        _vina_dock(monkeypatch, fake())

    assert len(caplog.records) == 1, caplog.text
    assert "[docking]" in caplog.text, caplog.text


# ── the temporary pose file ─────────────────────────────────────────────────

@pytest.mark.parametrize("outcome", [
    pytest.param(lambda: VinaRun(returncode=0, body=""), id="no-poses"),
    pytest.param(lambda: VinaRun(returncode=1), id="exit-code"),
    pytest.param(lambda: VinaRun(raises=OSError("no binary")), id="cannot-start"),
])
def test_the_pose_file_is_removed_whatever_vina_did(monkeypatch, outcome):
    fake = outcome()

    _vina_dock(monkeypatch, fake)

    assert fake.pose_files, fake.commands
    for path in fake.pose_files:
        assert not os.path.exists(path), path


def test_a_pose_file_that_refuses_to_be_deleted_is_not_a_failure(monkeypatch, caplog):
    """Windows holds a file open now and then; the poses are still valid."""
    fake = VinaRun(returncode=0,
                   body="REMARK VINA RESULT:    -8.4      0.000  0.000\n")
    attempted = []

    def refuse(path):
        attempted.append(path)
        raise OSError("device busy")

    monkeypatch.setattr(dk.subprocess, "run", fake)
    monkeypatch.setattr(dk.os, "unlink", refuse)
    with caplog.at_level("DEBUG", logger="biosuite.core.docking"):
        poses, reason = _vina_dock(monkeypatch, fake)

    assert len(attempted) == 1, "the pose file was never handed back"
    assert poses and reason == "", (poses, reason)
    assert "could not be removed" in caplog.text, caplog.text


def test_the_command_line_is_the_box_and_pose_count_it_was_given(monkeypatch):
    fake = VinaRun(returncode=0, body="REMARK VINA RESULT: -8.4\n")

    _vina_dock(monkeypatch, fake)

    cmd, kwargs = fake.commands[0]
    assert cmd[0] == 'vina'
    assert cmd[cmd.index('--receptor') + 1] == "receptor.pdbqt"
    assert cmd[cmd.index('--ligand') + 1] == "ligand.pdbqt"
    assert [float(cmd[cmd.index(n) + 1]) for n in
            ('--size_x', '--size_y', '--size_z')] == [16.0, 16.0, 16.0]
    assert cmd[cmd.index('--num_modes') + 1] == "3"
    assert kwargs['timeout'] == 3600
    assert kwargs['capture_output'] is True and kwargs['text'] is True


# ── what the user of dock() is told ─────────────────────────────────────────

def test_the_tool_check_answers_one_boolean_key_per_engine_it_can_use():
    """`dock()` reads `tools['vina']`; a dict without that key is a KeyError run."""
    tools = dk.check_docking_tools()

    assert set(tools) == {'vina'}, tools
    assert isinstance(tools['vina'], bool), tools


def test_a_vina_that_was_found_and_failed_is_not_reported_as_absent(
        monkeypatch, receptor, ligand):
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': True})
    monkeypatch.setattr(dk.subprocess, "run",
                        VinaRun(returncode=1, stderr="ERROR: bad receptor file"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dk.dock(receptor, ligand, center=[0, 0, 0], tool='vina')

    text = str(caught[0].message)
    assert "was tried but produced no usable poses" in text, text
    assert "not found" not in text, text
    assert "vina exited with code 1: ERROR: bad receptor file" in text, text
    assert result.engine == 'builtin', result


def test_an_installed_vina_that_wrote_nothing_says_that_instead(monkeypatch,
                                                                receptor, ligand):
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': True})
    monkeypatch.setattr(dk.subprocess, "run", VinaRun(returncode=0, body=""))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dk.dock(receptor, ligand, center=[0, 0, 0], tool='auto')

    assert "REMARK VINA RESULT" in str(caught[0].message), str(caught[0].message)


def test_a_machine_without_vina_is_still_told_to_install_it(monkeypatch,
                                                            receptor, ligand):
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': False})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dk.dock(receptor, ligand, center=[0, 0, 0], tool='auto')

    text = str(caught[0].message)
    assert "AutoDock Vina not found" in text, text
    assert "was tried" not in text, text


def test_the_substituted_result_does_not_claim_to_be_a_vina_result(monkeypatch,
                                                                   receptor, ligand):
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': True})
    monkeypatch.setattr(dk.subprocess, "run", VinaRun(returncode=1, stderr="no"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dk.dock(receptor, ligand, center=[0, 0, 0], tool='vina')

    assert result.engine == 'builtin'
    assert "arbitrary units, not kcal/mol" in result.message, result.message
