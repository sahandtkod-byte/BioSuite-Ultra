"""Temp-file safety and output-check regression tests (NEW-18, NEW-19).

NEW-18: 13 call sites used ``tempfile.mktemp()``, which only predicts a free
name.  Between the prediction and the open, anything on a shared /tmp can
create that path - typically a symlink to a file the process may write.

NEW-19: replacing it with ``secure_temp_path`` (which reserves the path by
*creating* it) turned every ``os.path.exists(out)`` success check into a
tautology, so an empty file would have been handed to a parser as a
successful tool run.  ``wrote_output`` is the correct check.
"""
import ast
import os
import pathlib

import pytest

from biosuite.core.utils import secure_temp_path, wrote_output

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / "biosuite"

# Windows does not model POSIX permission bits: os.chmod() there only toggles
# the read-only flag, and a freshly created file reports 0o666 no matter which
# mode was requested.  The private-mode guarantee is therefore asserted on the
# platform that enforces it (CI runs ubuntu-latest) instead of being watered
# down to an assertion that could never fail.
POSIX_MODES = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX mode bits are emulated on Windows; 0o600 is not observable")


# ── secure_temp_path ────────────────────────────────────────────────────────

def test_the_path_is_reserved_not_merely_predicted():
    path = secure_temp_path()
    try:
        assert os.path.exists(path), "mktemp's race is only closed by creating the file"
    finally:
        os.unlink(path)


@POSIX_MODES
def test_the_file_is_created_private():
    path = secure_temp_path()
    try:
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    finally:
        os.unlink(path)


def test_paths_are_unique():
    paths = [secure_temp_path() for _ in range(50)]
    try:
        assert len(set(paths)) == 50
    finally:
        for p in paths:
            os.unlink(p)


def test_suffix_and_prefix_are_honoured():
    path = secure_temp_path(suffix=".fasta", prefix="unit_")
    try:
        assert isinstance(path, str), "callers build on it with os.path, not with /"
        assert path.endswith(".fasta")
        assert os.path.basename(path).startswith("unit_")
    finally:
        os.unlink(path)


def test_the_reserved_file_starts_empty():
    path = secure_temp_path()
    try:
        assert os.path.getsize(path) == 0
    finally:
        os.unlink(path)


# ── wrote_output ────────────────────────────────────────────────────────────

def test_an_empty_reserved_file_is_not_output():
    """Exactly the NEW-19 tautology: it exists, but nothing was written."""
    path = secure_temp_path()
    try:
        assert os.path.exists(path)
        assert wrote_output(path) is False
    finally:
        os.unlink(path)


def test_a_non_empty_file_is_output():
    path = secure_temp_path()
    try:
        with open(path, "w") as handle:
            handle.write(">seq1\nACGT\n")
        assert wrote_output(path) is True
    finally:
        os.unlink(path)


def test_a_missing_path_is_not_output():
    path = secure_temp_path()
    os.unlink(path)
    assert wrote_output(path) is False


@pytest.mark.parametrize("bad", ["\x00", "", 123, b"\x00bytes"])
def test_a_malformed_path_is_not_output(bad):
    """`os.path.isfile` swallows the `ValueError` an embedded null raises in the
    stat itself — measured, `isfile('\\x00')` is False — so `wrote_output`'s own
    guard only has to cover the `OSError` race.  `123` is read as a file
    descriptor and answers False for the same reason."""
    assert wrote_output(bad) is False


def test_an_object_that_is_not_a_path_is_a_caller_bug_and_says_so():
    """The old code answered `False`, which sent the caller down its "the tool
    produced nothing" branch: our plumbing error became a report that blamed SPAdes
    or muscle.  Every real call site passes `secure_temp_path()`'s own return value,
    so a non-path can only be a programming error.  `os.path.isfile` already raises
    `TypeError` on it; the guard here does not catch that."""
    with pytest.raises(TypeError):
        wrote_output(None)


def test_a_file_deleted_between_the_two_stats_is_no_output(tmp_path, monkeypatch):
    """A reserved name is not a claim on the filesystem: a temp sweep can delete the
    file after `isfile()` has already said yes."""
    path = str(tmp_path / "out.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("chrom\tpos\n")

    def gone(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os.path, "getsize", gone)
    assert wrote_output(path) is False


def test_a_directory_at_the_path_is_not_output(tmp_path):
    """Measured on this machine: `getsize()` of a directory returns 0, so the old
    `exists()` + `getsize() > 0` answered False by a coincidence of the platform's
    stat semantics rather than by any rule.  `isfile()` makes it a rule: only a file
    that holds bytes counts as output, whatever the directory reports."""
    directory = tmp_path / "out"
    directory.mkdir()
    assert wrote_output(str(directory)) is False


# ── static guards against reintroducing the patterns ────────────────────────

def _python_files():
    return [p for p in PACKAGE.rglob("*.py")]


def test_no_module_calls_tempfile_mktemp():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "mktemp"):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, f"tempfile.mktemp() is a TOCTOU race: {offenders}"


def test_no_success_check_uses_bare_exists_on_a_reserved_path():
    """`if returncode == 0 and os.path.exists(out)` is always true now.

    Only variables actually assigned from secure_temp_path() are flagged:
    os.path.exists() on a user-supplied file or an mkdtemp() directory is a
    legitimate check, and wrote_output's own implementation must not match.
    """
    offenders = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        if "secure_temp_path" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            reserved = {
                target.id
                for node in ast.walk(func)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "secure_temp_path"
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if not reserved:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
                    continue
                for value in node.values:
                    if (isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Attribute)
                            and value.func.attr == "exists"
                            and value.args
                            and isinstance(value.args[0], ast.Name)
                            and value.args[0].id in reserved):
                        offenders.append(
                            f"{path.relative_to(REPO)}:{value.lineno}")
    assert not offenders, (
        "use wrote_output(); exists() is a tautology for a reserved path: "
        f"{offenders}")


def test_msa_has_no_silently_swallowed_exceptions():
    tree = ast.parse((PACKAGE / "core" / "msa.py").read_text(encoding="utf-8"))
    offenders = [h.lineno for h in ast.walk(tree)
                 if isinstance(h, ast.ExceptHandler)
                 and len(h.body) == 1 and isinstance(h.body[0], ast.Pass)]
    assert not offenders, f"except: pass hides tool failures at lines {offenders}"
