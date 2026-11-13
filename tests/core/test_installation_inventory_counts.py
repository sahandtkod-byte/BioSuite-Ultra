"""User-visible counts of this installation have to be measured, never remembered.

The CLI's HTML report (menu option 94) used to print ``Modules: 48`` and
``Tests: 203``.  The package shipped 47 modules when this was written, and the 203
was a test count from a release nobody re-counted — the report simply asserted both.
These tests pin the inventory helpers in :mod:`biosuite.core.utils` to an independent
walk of the package, pin the file counter to a fixture tree, and pin option 94 to the
helpers so a hand-written number cannot come back.
"""
import builtins
import pkgutil
import re
from pathlib import Path

from biosuite.core.utils import count_analysis_modules, count_test_files

REPO_ROOT = Path(__file__).resolve().parents[2]


def _modules_through_pkgutil() -> set:
    """Every importable module under ``biosuite.core``, found by walking packages.

    Deliberately not the same mechanism as the helper (which globs ``*.py``): two
    independent routes to the same number is what makes the number trustworthy.
    """
    import biosuite.core as core_pkg

    root = Path(core_pkg.__file__).parent

    def walk(directory: Path, prefix: str) -> set:
        found = set()
        for module in pkgutil.iter_modules([str(directory)]):
            if module.ispkg:
                found |= walk(directory / module.name, f"{prefix}{module.name}.")
            else:
                found.add(f"{prefix}{module.name}")
        return found

    return walk(root, "")


def test_count_analysis_modules_matches_a_pkgutil_walk():
    assert count_analysis_modules() == len(_modules_through_pkgutil())


def test_count_analysis_modules_names_the_modules_it_counts():
    """The count is only honest if the set behind it is the analysis package."""
    counted = _modules_through_pkgutil()
    assert {"sequence", "alignment", "cloning"} <= counted
    assert "workflow.pipeline" in counted
    assert "__init__" not in counted


def test_count_test_files_counts_only_test_modules(tmp_path, monkeypatch):
    (tmp_path / "tests" / "nested").mkdir(parents=True)
    (tmp_path / "tests" / "test_alpha.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "nested" / "test_beta.py").write_text("", encoding="utf-8")
    for decoy in ("helpers.py", "conftest.py", "test_notes.txt", "test_greek.md"):
        (tmp_path / "tests" / decoy).write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert count_test_files() == 2


def test_count_test_files_is_zero_outside_a_source_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert count_test_files() == 0


def test_count_test_files_matches_the_checkout_of_this_repo(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert count_test_files() == len(list((REPO_ROOT / "tests").rglob("test_*.py")))


def test_cli_report_prints_measured_counts(tmp_path, monkeypatch, capsys):
    """Option 94 must write the helpers' numbers into the report it saves."""
    from biosuite.cli import menu

    (tmp_path / "tests").mkdir()
    for name in ("test_one.py", "test_two.py", "test_three.py"):
        (tmp_path / "tests" / name).write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", ""])          # default title, default filename
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    menu._handle_menu_choice("94")

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    # report.py emits value first, then label, so the pairs have to be flipped back.
    shown = {label: value for value, label in
             re.findall(r'stat-value">([^<]*)</div>\s*<div class="stat-label">([^<]*)<', html)}
    assert shown == {"Modules": str(count_analysis_modules()), "Test files": "3"}


def test_cli_report_source_has_no_hard_coded_counts():
    """Guard against the literal coming back instead of the measurement."""
    from biosuite.cli import menu

    source = Path(menu.__file__).read_text(encoding="utf-8")
    block = source[source.index("elif choice == '94':"):source.index("elif choice == '95':")]
    assert "count_analysis_modules()" in block
    assert "count_test_files()" in block
    assert not re.search(r"add_stats\(\{[^}]*\d", block), block
