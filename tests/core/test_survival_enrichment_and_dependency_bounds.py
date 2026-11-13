"""Audit-remediation regression tests — Phase 6.

N13 is the Cox median split in ``biosuite/core/survival.py``: a split that could
empty one group, and a hazard ratio computed from medians that were never observed.

N14 is ``HTMLReport`` in ``biosuite/core/workflow/report.py``: report data went
into the document unescaped, so a section title or stat value carrying markup was
written out as markup.
"""
import pathlib

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

from biosuite.core.survival import cox_ph_summary, kaplan_meier

pytest.importorskip("scipy")

TIMES = np.array([10.0, 20.0, 30.0, 40.0])
EVENTS = np.array([1, 1, 1, 1])


def test_cox_ph_summary_accepts_plain_lists():
    """kaplan_meier always took lists; this one raised TypeError (N13)."""
    out = cox_ph_summary([10.0, 20.0, 30.0, 40.0], [1, 1, 1, 1], [1.0, 2.0, 2.0, 3.0])
    assert out['hazard_ratio'] == 0.5
    assert out['median_high_group'] == 40.0
    assert out['median_low_group'] == 20.0
    assert out['split'] == 'median'
    assert 0.0 <= out['p_value'] <= 1.0


def test_a_median_that_equals_the_mode_falls_back_to_a_rank_split():
    """[1,2,2,2]: "greater than the median" leaves the high group empty.

    The old code then divided by an infinite median and reported a hazard ratio
    of 0.0 — zero risk, from four patients and no such thing.
    """
    out = cox_ph_summary(TIMES, EVENTS, [1.0, 2.0, 2.0, 2.0])

    assert out['split'] == 'rank'
    assert out['hazard_ratio'] not in (0.0, None)
    assert np.isfinite(out['hazard_ratio'])
    assert out['message'] is None


def test_an_unobserved_median_yields_no_ratio_instead_of_nan_or_inf():
    """No events at all: both group medians are undefined, so no estimate exists."""
    out = cox_ph_summary(TIMES, [0, 0, 0, 0], [1.0, 2.0, 2.0, 3.0])

    assert out['hazard_ratio'] is None
    assert out['median_high_group'] is None
    assert out['median_low_group'] is None
    assert 'undefined' in out['message']
    assert out['p_value'] == 1.0


def test_the_ratio_is_the_low_group_median_survival_over_the_high_group_s():
    """For a well-behaved covariate the value must equal an independent calculation —
    and the calculation it agrees with is the module's own Kaplan-Meier median, not
    the median of the observed event times (which ignores censoring altogether)."""
    rng = np.random.default_rng(5)
    covariates = rng.uniform(0, 1, 60)
    times = np.where(covariates > 0.5, rng.exponential(5, 60), rng.exponential(30, 60))
    events = np.ones(60)

    out = cox_ph_summary(times, events, covariates)
    high = covariates > np.median(covariates)

    assert out['split'] == 'median'
    assert out['median_high_group'] == round(
        kaplan_meier(times[high], events[high]).median_survival, 1)
    assert out['median_low_group'] == round(
        kaplan_meier(times[~high], events[~high]).median_survival, 1)
    assert out['hazard_ratio'] == round(
        kaplan_meier(times[~high], events[~high]).median_survival
        / kaplan_meier(times[high], events[high]).median_survival, 3)


def test_a_constant_covariate_still_short_circuits():
    """Unchanged contract: nothing to split on is reported, not estimated — but now
    with the full key set, so a consumer reading any key gets None instead of a
    KeyError (measured before: ``out['median_high_group']`` raised)."""
    out = cox_ph_summary([1, 2, 3], [1, 1, 1], [5, 5, 5])

    assert out['hazard_ratio'] is None
    assert out['p_value'] == 1.0
    assert out['median_high_group'] is None and out['median_low_group'] is None
    assert out['split'] is None
    assert out['message'].startswith('Insufficient data')
    assert set(out) == set(cox_ph_summary(TIMES, EVENTS, [1.0, 2.0, 2.0, 3.0]))


# ── N14: HTMLReport must escape data and keep its own markup ────────────────

SCRIPT = "<script>alert('xss')</script>"
IMG = '<img src=x onerror="bad()">'
QUOTE_BREAK = '"><b>bold</b>'


def _injected_report():
    from biosuite.core.workflow.report import HTMLReport

    report = HTMLReport(SCRIPT, IMG)
    report.add_stats({"gene": QUOTE_BREAK})
    report.add_text(SCRIPT)
    report.add_error(IMG)
    report.add_success(QUOTE_BREAK)
    return report.to_html()


def test_report_data_never_reaches_the_document_as_markup():
    """Regression guard for N14: the payloads must arrive escaped."""
    import html

    document = _injected_report()
    for payload in (SCRIPT, IMG, QUOTE_BREAK):
        assert payload not in document, payload
        assert html.escape(payload, quote=True) in document, payload


def test_report_attribute_contexts_cannot_be_broken_out_of():
    """Titles also become anchor ids, which is an attribute context."""
    document = _injected_report()
    assert ' onmouseover="' not in document
    assert '" onmouseover' not in document
    assert '<b>bold</b>' not in document


def test_benign_report_content_still_renders():
    """Positive control: escaping must not flatten ordinary reports.

    Without this, a change that HTML-escapes the whole document — or one that
    writes nothing at all — would satisfy the two tests above.
    """
    import pandas as pd

    from biosuite.core.workflow.report import HTMLReport

    report = HTMLReport("Volcano plot", "RNA-seq run 12")
    report.add_section("Results", "<em>two up, one down</em>")
    report.add_table(pd.DataFrame({"gene": ["actin"], "log2fc": [1.5]}), "Table")
    report.add_stats({"Genes": 2})
    document = report.to_html()

    assert "<h1>Volcano plot</h1>" in document
    assert "<em>two up, one down</em>" in document
    assert "<table" in document and "actin" in document
    assert ">2<" in document


def test_add_plot_escapes_the_alt_attribute_and_still_embeds_the_image():
    import matplotlib.pyplot as plt

    from biosuite.core.workflow.report import HTMLReport

    figure, _ax = plt.subplots()
    try:
        document = HTMLReport("R").add_plot(figure, title=IMG, caption=QUOTE_BREAK).to_html()
    finally:
        plt.close(figure)

    assert IMG not in document
    assert "data:image/png;base64," in document
    assert "<em>" + QUOTE_BREAK not in document


def test_slug_keeps_ids_url_safe_and_never_empty():
    from biosuite.core.workflow.report import _slug

    assert _slug(IMG) == "img-src-x-onerror--bad"
    assert _slug("Volcano Plot") == "volcano-plot"
    assert _slug("<><>") == "section"
    assert _slug("") == "section"


# ── N18: a missing backend must not look like a biological result ───────────

def test_a_missing_backend_is_distinguishable_from_an_empty_result():
    """Regression guard for N18's silent-failure half.

    ``run_ora`` used to answer an absent goatools with the same empty report it
    would answer a genuinely unenriched gene list with.
    """
    from biosuite.core import enrichment as en

    rep = en.run_ora([], organism="human")
    assert rep.num_significant == 0
    assert rep.engine == ("goatools" if en.HAS_GOATOOLS else "none")


def test_enrichment_report_reports_the_backend_it_was_built_for():
    from biosuite.core.enrichment import EnrichmentReport, HAS_GOATOOLS

    implicit = EnrichmentReport("ORA", 3, 0)
    assert implicit.engine == ("goatools" if HAS_GOATOOLS else "none")
    assert EnrichmentReport("GSEA", 3, 0, engine="gseapy").engine == "gseapy"


# ── N17: an external-tool attempt must not leave a work directory behind ────

def _fake_ml_run(monkeypatch, tmp_path, *, tools, handler, method):
    """Drive build_tree with a faked engine; return the new temp entries."""
    import os
    import tempfile
    import warnings

    from biosuite.core import ml_phylogeny as mp

    alignment = tmp_path / "aln.fasta"
    alignment.write_text(">s1\nACGTACGT\n>s2\nACGTACGA\n>s3\nACGAACGT\n")
    monkeypatch.setattr(mp, "check_phylo_tools",
                        lambda: dict(tools, raxml_legacy=False))
    monkeypatch.setattr(mp.subprocess, "run", handler)
    before = set(os.listdir(tempfile.gettempdir()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mp.build_tree(str(alignment), method=method)
    added = set(os.listdir(tempfile.gettempdir())) - before
    return result, added, caught


def _proc(returncode):
    class Proc:
        pass
    proc = Proc()
    proc.returncode, proc.stdout, proc.stderr = returncode, "", ""
    return proc


def _leftover_dirs(added):
    import os
    import tempfile

    return [name for name in added
            if os.path.isdir(os.path.join(tempfile.gettempdir(), name))]


def test_a_successful_ml_run_leaves_no_work_directory_behind(tmp_path, monkeypatch):
    """Regression guard for N17.

    The engine output is read into ``newick_tree``, so the temporary work
    directory has no reason to survive the call — and it used to survive exactly
    the path that succeeded.
    """
    from biosuite.core import ml_phylogeny as mp

    def handler(cmd, **kwargs):
        prefix = cmd[cmd.index("--prefix") + 1]
        with open(prefix + ".raxml.bestTree", "w", encoding="utf-8") as handle:
            handle.write("((s1,s2)s3);\n")
        return _proc(0)

    result, added, _caught = _fake_ml_run(
        monkeypatch, tmp_path, method="raxml",
        tools={"raxml": True, "iqtree": False}, handler=handler)

    assert result.engine == "raxml"
    assert result.newick_tree.startswith("(")
    assert not result.tree_file, "a path into a deleted directory is not a result"
    assert _leftover_dirs(added) == [], added


def test_a_failing_ml_run_leaves_no_work_directory_either(tmp_path, monkeypatch):
    from biosuite.core import ml_phylogeny as mp

    result, added, caught = _fake_ml_run(
        monkeypatch, tmp_path, method="raxml",
        tools={"raxml": True, "iqtree": False},
        handler=lambda cmd, **kwargs: _proc(7))

    assert result.engine == "builtin"
    assert any("maximum likelihood" in str(w.message) for w in caught)
    assert _leftover_dirs(added) == [], added


# ── P2 dependency decision: the GUI extra must cover what the GUI imports ───

def _dependency_sets():
    """Core requirements and the `gui` extra, read without a TOML parser (CI is 3.10)."""
    import re

    core = {re.split(r"[\[<>=!;]", line.split("#")[0].strip())[0].strip()
            for line in (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")}
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^gui = \[(.*?)\n\]", text, re.S | re.M)
    gui = {re.split(r"[\[<>=!;]", item.strip('" ').strip())[0].strip()
           for item in match.group(1).splitlines() if item.strip().strip('",')} \
        if match else set()
    return core, gui


def _import_time_third_party(package_dir):
    """Third-party names imported at module or class level (executed on import)."""
    import ast
    import sys

    found = set()
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        stack = [tree]
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                       # deferred: optional by design
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Import):
                    found.update(a.name.split(".")[0] for a in child.names)
                elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                    found.add(child.module.split(".")[0])
                else:
                    stack.append(child)
    return {name for name in found
            if name not in sys.stdlib_module_names and name != "biosuite"}


def test_the_gui_extra_covers_every_import_time_gui_dependency():
    """The audit's P2 remedy was to uncomment customtkinter in requirements.txt.

    Measured instead: it is already declared in the pyproject ``gui`` extra, and
    requirements.txt documents its trailing optional blocks as reference-only, so
    uncommenting it would hand a GUI dependency to every headless install.  What is
    worth pinning is the invariant the remedy was reaching for.
    """
    core, gui = _dependency_sets()
    assert "customtkinter" in gui, sorted(gui)
    needed = _import_time_third_party(_REPO_ROOT / "biosuite" / "gui")
    missing = sorted(needed - core - gui)
    assert not missing, f"biosuite.gui needs {missing}, which no extra installs"
