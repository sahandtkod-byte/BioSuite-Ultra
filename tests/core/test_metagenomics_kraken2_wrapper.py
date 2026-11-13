"""The Kraken2 wrapper of metagenomics: its command line, its reasons, its files.

Nothing here runs a real kraken2 — ``subprocess.run`` is intercepted, so the tests
assert on the argv that would have been sent and on the reason the wrapper hands back.
The reason matters because a bare ``None`` used to make a kraken2 that was installed
and then failed indistinguishable from one that is not installed, and
``classify_reads()`` then told the user to install a program they had installed.
"""
import os
import subprocess

import pytest

from biosuite.core import metagenomics as mg

CLASSIFIED_LINE = "C\tread1\t561\t50\t561:0 561:1\n"


class FakeRun:
    """Stands in for subprocess.run; records the argv and may produce a file."""

    def __init__(self, writes=None, returncode=0, stdout="", stderr="", raises=None):
        self.calls = []
        self.writes = writes or {}
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if self.raises is not None:
            raise self.raises
        out = cmd[cmd.index("--output") + 1] if "--output" in cmd else None
        if out is not None and self.returncode == 0 and self.writes:
            with open(out, "w", encoding="utf-8") as handle:
                handle.write(self.writes.get("body", ""))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def reads_file(tmp_path):
    path = tmp_path / "reads.fastq"
    path.write_text("@r1\nATCGATCGATCGATCG\n+\nIIII\n", newline="")
    return str(path)


def test_the_command_line_is_the_documented_one(tmp_path, monkeypatch, reads_file):
    out = tmp_path / "kraken.out"
    fake = FakeRun(returncode=0, writes={"body": CLASSIFIED_LINE})
    monkeypatch.setattr(mg.subprocess, "run", fake)
    result, failure = mg._kraken2_classify(reads_file, database="/db",
                                           output_file=str(out))
    assert fake.calls == [["kraken2", "--threads", "1", "--output", str(out),
                           "--db", "/db", reads_file]]
    assert failure == ""
    assert result.engine == "kraken2"


def test_the_database_flag_is_only_added_when_one_is_named(tmp_path, monkeypatch,
                                                           reads_file):
    out = tmp_path / "kraken.out"
    out.write_text(CLASSIFIED_LINE, encoding="utf-8")
    fake = FakeRun(returncode=0, writes={"body": CLASSIFIED_LINE})
    monkeypatch.setattr(mg.subprocess, "run", fake)
    mg._kraken2_classify(reads_file, database=None, output_file=str(out))
    assert "--db" not in fake.calls[0]


def test_a_nonzero_exit_becomes_a_reason_naming_the_tools_last_line(
        monkeypatch, reads_file):
    fake = FakeRun(returncode=3, stderr="kraken2: database /db not found")
    monkeypatch.setattr(mg.subprocess, "run", fake)
    result, failure = mg._kraken2_classify(reads_file, "/db", "never.txt")
    assert result is None
    assert failure == "it exited with 3: kraken2: database /db not found"


def test_a_nonzero_exit_with_no_output_says_so(monkeypatch, reads_file):
    monkeypatch.setattr(mg.subprocess, "run", FakeRun(returncode=2))
    result, failure = mg._kraken2_classify(reads_file, "/db", "never.txt")
    assert result is None
    assert failure == "it exited with 2: no output"


@pytest.mark.parametrize("crash", [
    pytest.param(OSError(2, "No such file or directory"), id="missing-binary"),
    pytest.param(subprocess.TimeoutExpired(cmd="kraken2", timeout=7200), id="timeout"),
])
def test_a_launch_failure_names_the_error(monkeypatch, reads_file, crash):
    """'Kraken2 not found' was the message for both an absent binary and a kraken2
    that could not start; only the first of those is true."""
    monkeypatch.setattr(mg.subprocess, "run", FakeRun(raises=crash))
    result, failure = mg._kraken2_classify(reads_file, "/db", "never.txt")
    assert result is None
    assert failure.startswith("it could not be executed:")
    assert str(crash) in failure


def test_an_exit_zero_run_that_wrote_nothing_is_reported_as_such(
        tmp_path, monkeypatch, reads_file):
    """secure_temp_path reserves the file, so exists() alone would be a tautology;
    the emptiness check is what catches a tool that produced nothing."""
    out = tmp_path / "empty.out"          # never written by the fake
    monkeypatch.setattr(mg.subprocess, "run", FakeRun(returncode=0, writes={}))
    result, failure = mg._kraken2_classify(reads_file, "/db", str(out))
    assert result is None
    assert failure == "it exited cleanly but wrote nothing to its report"


def test_a_report_that_cannot_be_decoded_is_a_reason_not_a_crash(
        tmp_path, monkeypatch, reads_file):
    out = tmp_path / "binary.out"

    def write_bytes(cmd, **kwargs):
        with open(cmd[cmd.index("--output") + 1], "wb") as handle:
            # 0xff is not the start of any UTF-8 sequence, so the reader cannot
            # even reach a field boundary.
            handle.write(b"\xff\xfe\x00not a report\x80")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mg.subprocess, "run", write_bytes)
    result, failure = mg._kraken2_classify(reads_file, "/db", str(out))
    assert result is None
    assert failure.startswith("its report could not be read:")


def test_a_failed_run_does_not_leave_its_reserved_temp_file(tmp_path, monkeypatch,
                                                           reads_file):
    """The wrapper reserves its own output path; leaking it on every failed run
    would grow the temp directory one file per call."""
    reserved = tmp_path / "reserved.out"
    monkeypatch.setattr(mg, "secure_temp_path", lambda *a, **k: str(reserved))
    monkeypatch.setattr(mg.subprocess, "run", FakeRun(returncode=3))
    result, failure = mg._kraken2_classify(reads_file, "/db")
    assert result is None
    assert not reserved.exists()


def test_a_successful_run_keeps_a_caller_named_output(tmp_path, monkeypatch, reads_file):
    """When the caller asked for a specific file it is theirs to keep."""
    out = tmp_path / "wanted.out"
    monkeypatch.setattr(mg.subprocess, "run",
                        FakeRun(returncode=0, writes={"body": CLASSIFIED_LINE}))
    mg._kraken2_classify(reads_file, "/db", str(out))
    assert out.exists()


def test_a_reserved_path_that_cannot_be_removed_is_logged(tmp_path, monkeypatch,
                                                          reads_file, caplog):
    """Cleaning up is not allowed to fail in silence: the file is outside the
    caller's view, so the log line is the only trace of it."""
    reserved = tmp_path / "stuck.out"
    monkeypatch.setattr(mg, "secure_temp_path", lambda *a, **k: str(reserved))
    monkeypatch.setattr(mg.subprocess, "run", FakeRun(returncode=0,
                                                      writes={"body": CLASSIFIED_LINE}))

    def refuse(*args, **kwargs):
        raise OSError(22, "in use")

    monkeypatch.setattr(mg.os, "remove", refuse)
    import logging
    with caplog.at_level(logging.WARNING):
        result, _ = mg._kraken2_classify(reads_file, "/db")
    assert result.engine == "kraken2"
    assert "Could not remove the reserved kraken2 report" in caplog.text


def test_the_reads_file_is_passed_through_unchanged(tmp_path, monkeypatch):
    """A path with a space must reach kraken2 as one argument, not as two."""
    reads = tmp_path / "my reads.fastq"
    reads.write_text("@r1\nATCG\n+\nIII\n", newline="")
    fake = FakeRun(returncode=1, stderr="no")
    monkeypatch.setattr(mg.subprocess, "run", fake)
    mg._kraken2_classify(str(reads), "/db", str(tmp_path / "out"))
    assert fake.calls[0][-1] == str(reads)
    assert os.path.basename(fake.calls[0][-1]) == "my reads.fastq"
