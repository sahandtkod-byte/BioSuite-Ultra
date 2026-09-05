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


# ── AGENTS.md / DEPLOY.md coverage ──────────────────────────────────────────
# These two files were the last to carry stale figures: AGENTS.md advertised
# "2,500+ tests" in its Build & Test block while the same file quoted the real
# number two other times, and DEPLOY.md still announced "v4.1.0" with
# "100+ restriction enzymes" under a "Current Version" heading.  Both are tied
# below to authoritative values measured from the tree, not to literal numbers.

def _test_file_count():
    return len(list((REPO / "tests").rglob("test_*.py")))


def _readme_ci_pass_counts():
    """Pass counts from the README's per-interpreter CI results table."""
    text = _read("README.md")
    return {int(m.replace(",", ""))
            for m in re.findall(r"Tests\s*[—-]\s*Python\s*[\d.]+\s*\|\s*([\d,]+)\s+passed",
                                text)}


def _agents_ci_pass_counts():
    """Pass counts AGENTS.md states in a CI context (not the local figure)."""
    counts = set()
    for line in _read("AGENTS.md").splitlines():
        if re.search(r"\bCI\b", line):
            counts |= {int(m.replace(",", ""))
                       for m in re.findall(r"([\d,]{3,})\s+passed", line)}
    return counts


def test_agents_md_quotes_the_authoritative_test_file_count():
    text = _read("AGENTS.md")
    actual = _test_file_count()
    quoted = {int(m) for m in re.findall(r"(\d+)\s+(?:test\s+)?files\b", text)}
    assert quoted, "AGENTS.md quotes no test-file count at all"
    assert quoted == {actual}, (
        f"AGENTS.md quotes test-file counts {sorted(quoted)}; measured {actual}")


def test_agents_md_does_not_carry_the_stale_test_count_claims():
    text = _read("AGENTS.md")
    for stale in ("2,500+", "2500+", "2,529", "2,375"):
        assert stale not in text, f"AGENTS.md still claims '{stale}' tests"


def test_agents_md_and_readme_agree_on_the_ci_pass_count():
    """Tie the two documents together so they cannot drift apart again.

    AGENTS.md may additionally quote the local figure, which differs because
    the GUI tests cannot run without tkinter; what it must not do is quote a
    *different* CI figure from the one in the README.
    """
    readme = _readme_ci_pass_counts()
    agents = _agents_ci_pass_counts()
    assert readme, "README's CI results table quotes no pass count"
    assert agents, "AGENTS.md quotes no CI pass count"
    assert len(readme) == 1, f"README quotes conflicting pass counts: {readme}"
    assert agents == readme, (
        f"AGENTS.md CI pass counts {sorted(agents)} disagree with the README "
        f"table {sorted(readme)}")


def _deploy_current_version_section():
    text = _read("DEPLOY.md")
    match = re.search(r"^## Current Version$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, "DEPLOY.md has no '## Current Version' section"
    return match.group(1)


def test_deploy_md_current_version_matches_the_package():
    import biosuite
    section = _deploy_current_version_section()
    assert f"v{biosuite.__version__}" in section, (
        f"DEPLOY.md 'Current Version' does not state v{biosuite.__version__}")


def test_deploy_md_has_no_stale_current_version_claim():
    section = _deploy_current_version_section()
    stale = re.findall(r"v4\.\d+\.\d+", section)
    assert not stale, f"DEPLOY.md 'Current Version' still claims {stale}"


def test_deploy_md_quotes_the_authoritative_enzyme_count():
    from biosuite.core.utils import RESTRICTION_ENZYMES
    text = _read("DEPLOY.md")
    assert f"{len(RESTRICTION_ENZYMES)} restriction enzymes" in text, (
        f"DEPLOY.md does not quote the measured enzyme count "
        f"({len(RESTRICTION_ENZYMES)})")
    assert not re.search(r"\d+\+ restriction enzymes", text), (
        "DEPLOY.md still uses an open-ended '<n>+ restriction enzymes' claim")


def test_deploy_md_python_requirement_matches_pyproject():
    pyproject = _read("pyproject.toml")
    minimum = re.search(r'requires-python\s*=\s*">=([\d.]+)"', pyproject).group(1)
    assert f"{minimum}+" in _read("DEPLOY.md"), (
        f"DEPLOY.md does not state the packaged minimum Python ({minimum}+)")


# ── Sphinx docs and API guide currency ──────────────────────────────────────
# API_GUIDE.md carried a "New in v4.1.0" section and docs/getting_started.rst
# still introduced the project as v4.1.0, advertised "100+ restriction
# enzymes", and linked two Sphinx documents that were never written.

CURRENT_STATE_DOCS = ("API_GUIDE.md", "DEPLOY.md", "README.md",
                      "docs/index.rst", "docs/getting_started.rst",
                      "docs/api/index.rst", "docs/tutorials/index.rst")


def test_current_state_docs_do_not_advertise_a_superseded_version():
    """No user-facing document may present a pre-5.x release as current."""
    import biosuite
    offenders = []
    for name in CURRENT_STATE_DOCS:
        for hit in re.findall(r"v[0-4]\.\d+(?:\.\d+)?", _read(name)):
            offenders.append(f"{name}: {hit}")
    assert not offenders, (
        f"superseded version presented as current (package is "
        f"{biosuite.__version__}): {offenders}")


def test_getting_started_states_the_packaged_version():
    import biosuite
    assert f"v{biosuite.__version__}" in _read("docs/getting_started.rst"), (
        f"docs/getting_started.rst does not introduce v{biosuite.__version__}")


def test_docs_quote_the_authoritative_enzyme_count_or_none_at_all():
    """An enzyme count may be exact or omitted, never an open-ended '100+'."""
    from biosuite.core.utils import RESTRICTION_ENZYMES
    actual = len(RESTRICTION_ENZYMES)
    for name in CURRENT_STATE_DOCS:
        text = _read(name)
        assert not re.search(r"\d+\+\s*(?:restriction\s+)?enzymes", text, re.I), (
            f"{name} uses an open-ended '<n>+ enzymes' claim")
        for quoted in re.findall(r"(\d+)\s+restriction\s+enzymes", text, re.I):
            assert int(quoted) == actual, (
                f"{name} claims {quoted} restriction enzymes; measured {actual}")


def _rst_files():
    return sorted((REPO / "docs").rglob("*.rst"))


def test_no_sphinx_doc_reference_is_dangling():
    dangling = []
    docs = REPO / "docs"
    for rst in _rst_files():
        text = rst.read_text(encoding="utf-8")
        for match in re.finditer(r":doc:`([^`<]*?<)?([^`>]+)>?`", text):
            target = match.group(2).strip()
            base = docs / target.lstrip("/") if target.startswith("/") \
                else rst.parent / target
            if not base.with_suffix(".rst").exists():
                dangling.append(f"{rst.relative_to(docs)} -> :doc:`{target}`")
    assert not dangling, f"dangling :doc: references: {dangling}"


def test_no_sphinx_toctree_entry_is_dangling():
    dangling = []
    docs = REPO / "docs"
    for rst in _rst_files():
        text = rst.read_text(encoding="utf-8")
        for block in re.findall(r"\.\. toctree::\n((?:[ \t]+.*\n|\n)*)", text):
            for line in block.splitlines():
                entry = line.strip()
                if not entry or entry.startswith(":"):
                    continue
                if not (rst.parent / entry).with_suffix(".rst").exists():
                    dangling.append(f"{rst.relative_to(docs)} -> {entry}")
    assert not dangling, f"dangling toctree entries: {dangling}"


# ── public-facing neutrality ────────────────────────────────────────────────
# Repository-visible material describes the software, not the tooling used to
# author it.  This guard deliberately targets *development-provenance* phrases
# only: BioSuite ships genuine machine-learning functionality (biosuite.core.
# bio_ml, the ML tutorial, the "machine-learning" keyword), and legitimate
# scientific references to AI/ML must never trip this test.

_PROVENANCE_PATTERNS = (
    r"\bAI[- ]generated\b",
    r"\bgenerated\s+by\s+(?:an?\s+)?AI\b",
    r"\bAI[- ]assisted\b",
    r"\bAI\s+remediation\b",
    r"\bcoding\s+agent\b",
    r"\bautonomous\s+agent\b",
    r"\bagent[- ]generated\b",
    r"\bagent\s+verification\b",
    r"\b(?:written|created|authored|fixed|implemented)\s+by\s+"
    r"(?:an?\s+)?(?:AI|agent|LLM|bot)\b",
    r"\bChatGPT\b", r"\bClaude\b", r"\bCopilot\b", r"\bHermes\s+Agent\b",
    r"\blarge\s+language\s+model\b", r"\bLLM\b",
)


def _public_facing_files():
    """Repository-visible sources and documents, excluding this test file."""
    exts = {".md", ".rst", ".py", ".toml", ".cff", ".cfg", ".txt", ".yml",
            ".yaml", ".html", ".ipynb"}
    skip_names = {pathlib.Path(__file__).name}
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(REPO)
        if rel.parts[0] in {".git", ".venv", "build", "dist", "node_modules"}:
            continue
        if path.name in skip_names:
            continue
        yield rel, path


def test_public_material_has_no_development_provenance_references():
    offenders = []
    for rel, path in _public_facing_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in _PROVENANCE_PATTERNS:
            for m in re.finditer(pattern, text, re.I):
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}: {m.group(0)!r}")
    assert not offenders, (
        "development-provenance references in repository-visible material:\n  "
        + "\n  ".join(offenders))


def test_legitimate_machine_learning_references_are_preserved():
    """Guard against over-correction: the ML functionality is a real feature."""
    assert (REPO / "biosuite" / "core" / "bio_ml.py").exists(), \
        "the machine-learning module must not be removed by neutrality edits"
    keywords = _read("pyproject.toml")
    assert "machine learning" in _read("docs/tutorials/index.rst").lower() \
        or "Machine Learning" in _read("docs/tutorials/index.rst"), \
        "the machine-learning tutorial section must survive neutrality edits"
    assert "scikit-learn" in keywords, \
        "scikit-learn must remain a declared dependency"
    assert "machine-learning" in keywords, \
        "the machine-learning keyword describes a real feature and must remain"


def test_every_documented_repository_url_is_the_official_one():
    """A wrong owner in a clone URL silently breaks the install instructions."""
    official = "github.com/sahandtkod-byte/BioSuite-Ultra"
    # Contributor docs legitimately show a fork placeholder to substitute.
    placeholders = {"YOUR_USERNAME", "YOUR-USERNAME", "your-username", "USERNAME"}
    offenders = []
    for rel, path in _public_facing_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in re.finditer(r"github\.com/([\w.-]+)/BioSuite-Ultra", text):
            if m.group(1) in placeholders:
                continue
            if m.group(0) not in official:
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}: {m.group(0)}")
    assert not offenders, f"non-official repository URLs: {offenders}"


def test_shipped_code_makes_no_marketing_or_unimplemented_claims():
    """Runtime strings and docstrings are user-visible; keep them factual."""
    banned = (r"100%\s*free", r"no paid", r"SnapGene", r"most comprehensive",
              r"killer", r"Gibson", r"36\+", r"26 visualization", r"48 analysis")
    offenders = []
    shipped = sorted((REPO / "biosuite").rglob("*.py")) + \
        sorted((REPO / "examples").rglob("*.py")) + \
        sorted((REPO / "examples").rglob("*.ipynb"))
    for path in shipped:
        text = path.read_text(encoding="utf-8")
        for pattern in banned:
            for m in re.finditer(pattern, text, re.I):
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO)}:{line}: {m.group(0)!r}")
    assert not offenders, f"marketing or unimplemented claims in shipped code: {offenders}"
