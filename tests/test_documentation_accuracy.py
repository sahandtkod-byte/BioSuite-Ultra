"""Documentation-accuracy tests (BSU-024, BSU-025).

The repository's docs stated counts that were simply false for this tree:
117 CLI options (99), 40 REST endpoints (38), "1,444 tests in 30 files",
and AGENTS.md still advertised version 4.2.5.

These tests re-measure from the code, so the numbers cannot drift back
unnoticed. They assert the documented figure equals the measured one; when the
code legitimately grows, the docs must be updated in the same commit.
"""
import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(name):
    path = REPO / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return path.read_text(encoding="utf-8")


# ── measured ground truth ───────────────────────────────────────────────────

def _analysis_module_count():
    core = REPO / "biosuite" / "core"
    mods = [p for p in core.glob("*.py") if p.name != "__init__.py"]
    wf = [p for p in (core / "workflow").glob("*.py") if p.name != "__init__.py"]
    return len(mods) + len(wf)


def _plotting_function_counts():
    plot = REPO / "biosuite" / "plotting"
    public = total = 0
    for p in plot.glob("*.py"):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        total += len(fns)
        public += len([f for f in fns if not f.name.startswith("_")])
    return public, total


def _api_route_count():
    from biosuite.api import app
    return len([r for r in app.routes
                if getattr(r, "path", "").startswith("/api/")])


# ── version consistency (BSU-024) ───────────────────────────────────────────

def test_version_is_a_single_source_of_truth():
    import biosuite
    version = biosuite.__version__
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'version\s*=\s*["\']{re.escape(version)}["\']', pyproject), \
        "pyproject.toml disagrees with biosuite.__version__"


def test_agents_md_does_not_advertise_a_stale_version():
    import biosuite
    text = _read("AGENTS.md")
    assert "4.2.5" not in text.split("## ")[0] or biosuite.__version__ == "4.2.5"
    assert biosuite.__version__ in text


def test_citation_file_matches_the_package_version():
    import biosuite
    path = REPO / "CITATION.cff"
    if not path.exists():
        pytest.skip("no CITATION.cff")
    assert biosuite.__version__ in path.read_text(encoding="utf-8")


def test_the_api_app_reports_the_package_version():
    import biosuite
    from biosuite.api import app
    assert app.version == biosuite.__version__


# ── documented counts match the code (BSU-025) ──────────────────────────────

def test_no_document_still_claims_117_cli_options():
    for name in ("README.md", "AGENTS.md"):
        assert "117" not in _read(name), f"{name} still claims 117 CLI options"


def test_no_document_still_claims_1444_tests():
    for name in ("README.md", "AGENTS.md"):
        text = _read(name)
        assert "1,444" not in text and "1444" not in text


def test_no_document_still_claims_40_endpoints():
    for name in ("README.md", "AGENTS.md"):
        assert "40 endpoints" not in _read(name)


def test_documented_analysis_module_count_is_correct():
    measured = _analysis_module_count()
    assert measured == 47, f"module count changed to {measured}; update the docs"
    assert str(measured) in _read("README.md")


def test_documented_api_endpoint_count_is_correct():
    measured = _api_route_count()
    assert measured == 38, f"endpoint count changed to {measured}; update the docs"


def test_documented_plotting_function_count_is_correct():
    public, total = _plotting_function_counts()
    assert public == 105, f"public plotting functions changed to {public}"
    assert total == 123, f"total plotting functions changed to {total}"


def test_restriction_enzyme_count_is_correct():
    from biosuite.core.utils import RESTRICTION_ENZYMES
    assert len(RESTRICTION_ENZYMES) == 169


def test_gui_tab_count_is_correct():
    tabs = REPO / "biosuite" / "gui" / "tabs"
    count = len([p for p in tabs.glob("*.py") if p.name != "__init__.py"])
    assert count == 11, f"GUI tab count changed to {count}; update the docs"


def test_test_file_count_claim_is_not_wildly_wrong():
    actual = len(list((REPO / "tests").rglob("test_*.py")))
    assert actual > 100, f"only {actual} test files found"
    text = _read("AGENTS.md")
    assert "30 test files" not in text and "30 files" not in text
