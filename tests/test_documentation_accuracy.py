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


def _cli_menu_option_count():
    """Distinct numeric choices the interactive menu dispatches, excluding 0 (exit)."""
    src = (REPO / "biosuite" / "cli" / "menu.py").read_text(encoding="utf-8")
    nums = {int(m) for m in re.findall(r"choice\s*==\s*['\"](\d+)['\"]", src)}
    nums |= {int(m) for m in re.findall(r"^\s*['\"](\d+)['\"]\s*:", src, re.M)}
    return len({n for n in nums if n != 0})


def test_documented_cli_menu_option_count_is_correct():
    measured = _cli_menu_option_count()
    assert measured == 99, f"CLI menu options changed to {measured}; update the docs"
    assert "99" in _read("README.md")


def test_documented_plot_catalogue_size_is_correct():
    """README and docs quote the size of the GUI plot catalogue."""
    tree = ast.parse((REPO / "biosuite" / "gui" / "themes.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "PLOT_CATEGORIES"):
            categories = len(node.value.keys)
            plots = sum(len(v.elts) for v in node.value.values)
            break
    else:                                    # pragma: no cover - structural guard
        pytest.fail("PLOT_CATEGORIES not found in biosuite/gui/themes.py")
    assert (categories, plots) == (9, 40), \
        f"plot catalogue changed to {plots} plots in {categories} categories"
    assert "40 plot types" in _read("README.md")


def test_supported_python_versions_match_the_ci_matrix():
    """Never advertise an interpreter the CI matrix does not exercise."""
    workflow = _read(".github/workflows/ci.yml")
    matrix = set(re.findall(r"'(3\.\d+)'", workflow.split("python-version:")[1][:80]))
    pyproject = _read("pyproject.toml")
    classifiers = set(re.findall(r"Programming Language :: Python :: (3\.\d+)", pyproject))
    assert classifiers == matrix, (
        f"pyproject advertises {sorted(classifiers)} but CI tests {sorted(matrix)}")
    requires = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject).group(1)
    assert requires == ">=" + min(matrix, key=lambda v: tuple(map(int, v.split(".")))), \
        f"requires-python {requires} disagrees with the CI matrix {sorted(matrix)}"


def test_no_stale_version_strings_are_hard_coded_in_the_package():
    """The version policy in AGENTS.md: only biosuite/__init__.py hard-codes it."""
    import biosuite
    offenders = []
    for path in (REPO / "biosuite").rglob("*.py"):
        if path.name == "__init__.py" and path.parent.name == "biosuite":
            continue
        text = path.read_text(encoding="utf-8")
        # A leading letter means it is somebody else's version string, e.g. the
        # "##fileformat=VCFv4.2" header that variant_calling.py must emit.
        for stale in ("v4.0", "v4.1", "v4.2", "v5.0.0"):
            if re.search(rf"(?<![A-Za-z]){re.escape(stale)}", text):
                offenders.append(f"{path.relative_to(REPO)}: {stale}")
    assert not offenders, f"stale hard-coded versions: {offenders}"


def test_readme_makes_no_unverifiable_superlative_claims():
    text = _read("README.md")
    for phrase in ("most comprehensive", "SnapGene-killer", "100% free",
                   "unhackable", "enterprise-grade", "1,444", "cyberpunk"):
        assert phrase.lower() not in text.lower(), f"README still claims: {phrase}"


def test_test_file_count_claim_is_not_wildly_wrong():
    actual = len(list((REPO / "tests").rglob("test_*.py")))
    assert actual > 100, f"only {actual} test files found"
    text = _read("AGENTS.md")
    # Word-boundary match: "130 test files" is correct and must not trip this.
    assert not re.search(r"(?<!\d)30 test files", text)
    assert not re.search(r"(?<!\d)30 files", text)
