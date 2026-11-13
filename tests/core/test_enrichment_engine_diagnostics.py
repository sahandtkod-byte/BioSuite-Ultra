"""How enrichment.py decides which engine it has, and what it says when it has none.

The statsmodels compatibility shim, the goatools import and the gseapy import all sat
in one `try/except ImportError`, and every failure answered the same sentence: "not
installed. Run: pip install goatools".  A user whose goatools was installed but whose
dependency failed was therefore told to install the thing they had, and the reason the
interpreter actually gave was discarded without a trace.
"""
import contextlib
import importlib.util
import logging
import os
import sys
import types
import urllib.request

import pytest

from biosuite.core import enrichment as en


@contextlib.contextmanager
def executed_with(modules):
    """Run enrichment.py again with ``modules`` (import name -> module) injected.

    The HAS_* flags and the import reasons are decided while the module body runs, so
    the only way to reach the "engine present" and "engine half-broken" sides is to
    execute the file with the dependency substituted.
    """
    suffix = "_".join(sorted(name.replace(".", "_") for name in modules))
    alias = f"biosuite.core._enrichment_with_{suffix}"
    names = list(modules) + ["statsmodels", "statsmodels.sandbox",
                             "statsmodels.sandbox.stats"]
    parked = {name: sys.modules[name] for name in names if name in sys.modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(alias, en.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        sys.modules.pop(alias, None)
        sys.modules.update(parked)


def _goatools_modules(study=object, dag=object):
    goatools = types.ModuleType("goatools")
    goatools.__path__ = []
    go_enrichment = types.ModuleType("goatools.go_enrichment")
    go_enrichment.GOEnrichmentStudy = study
    obo_parser = types.ModuleType("goatools.obo_parser")
    obo_parser.GODag = dag
    goatools.go_enrichment, goatools.obo_parser = go_enrichment, obo_parser
    return {"goatools": goatools, "goatools.go_enrichment": go_enrichment,
            "goatools.obo_parser": obo_parser}


# ── why the engine is unusable is part of the answer ─────────────────────────

def test_an_engine_that_imports_but_is_broken_is_not_reported_as_absent():
    """`from goatools.go_enrichment import GOEnrichmentStudy` failing inside the
    package is an import failure, and the old text called it "not installed"."""
    modules = _goatools_modules()
    broken = modules["goatools.go_enrichment"]

    def refuse(name):
        raise ImportError("cannot import name 'x' from a goatools dependency")

    broken.__getattr__ = refuse
    with executed_with(modules) as module:
        assert module.HAS_GOATOOLS is False
        assert "cannot import name" in module.GOATOOLS_IMPORT_ERROR
        text = module._missing_engine_text("ORA")
        assert text.startswith("goatools is not usable. Run: pip install goatools")
        assert "cannot import name" in text, text


def test_a_dependency_of_the_shim_does_not_decide_whether_the_engine_exists():
    """The statsmodels sandbox path is a compatibility detail, not goatools.

    It used to sit in the same `try/except ImportError` as the engine import, so
    any failure there answered "goatools not installed".
    """
    modules = _goatools_modules()
    modules["statsmodels.sandbox.stats.multicomp"] = None      # import raises
    with executed_with(modules) as module:
        assert module.HAS_GOATOOLS is True
        assert module.GOATOOLS_IMPORT_ERROR == ""
        assert module._engine_for("ORA") == "goatools"


def test_an_absent_engine_reports_the_import_error_it_actually_got():
    """gseapy is genuinely not installed here, and that is what the reason says."""
    assert en.HAS_GSEAPY is False
    assert "gseapy" in en.GSEAPY_IMPORT_ERROR
    text = en._missing_engine_text("GSEA")
    assert text.startswith("gseapy is not usable. Run: pip install gseapy ("), text
    assert "not installed." not in text, text


def test_the_import_failure_is_logged_while_the_module_loads(caplog):
    """The reason has to reach the log, not only the message of one call."""
    with caplog.at_level(logging.DEBUG, logger="biosuite.core.enrichment"):
        with executed_with(_goatools_modules()):
            pass

    assert "gseapy is not usable" in caplog.text, caplog.text
    assert "No module named 'gseapy'" in caplog.text, caplog.text


# ── which backend an analysis type may claim ─────────────────────────────────

def test_an_unknown_analysis_type_is_given_no_backend():
    """`_engine_for` answered gseapy for every name that was not 'ORA'."""
    assert en._engine_for("reactome") == "none"
    assert en._engine_for("") == "none"
    assert en._engine_for("ORA") == ("goatools" if en.HAS_GOATOOLS else "none")
    assert en._engine_for("GSEA") == ("gseapy" if en.HAS_GSEAPY else "none")
    assert en._engine_for("KEGG") == ("gseapy" if en.HAS_GSEAPY else "none")


def test_a_report_built_with_an_unknown_analysis_type_claims_no_engine():
    assert en.EnrichmentReport("reactome", 10, 0).engine == "none"


# ── the ontology file the ORA run is built on ────────────────────────────────

def test_an_unusable_ontology_stops_before_the_engine_sees_a_none_path(monkeypatch):
    """`GODag(None)` used to be the answer, wrapped as "ORA analysis error".

    The download helper logs its failure and returns None; the caller handed that
    None straight to the parser and reported a Python complaint as the study.
    """
    touched = []

    class Study:
        def __init__(self, *args, **kwargs):
            touched.append("study")

        def run_study(self, genes):
            return []

    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GOEnrichmentStudy", Study, raising=False)
    monkeypatch.setattr(en, "GODag", lambda path: touched.append(path), raising=False)
    monkeypatch.setattr(en, "_download_go_obo", lambda: None)
    monkeypatch.setattr(en, "_load_associations_file",
                        lambda path: {"672": {"GO:0006915"}})

    report = en.run_ora(["672"], associations_file="ignored.tsv")

    assert report.message == ("No GO ontology available: go-basic.obo is not cached "
                              "and could not be downloaded. Pass obo_file=<path>."), \
        report.message
    assert report.results == [] and report.num_input_genes == 1
    assert touched == [], "neither the parser nor the study may run without an ontology"


def test_an_ontology_that_cannot_be_downloaded_is_a_warning_not_an_info_line(
        home_and_block_network, caplog):
    def refuse(*args, **kwargs):
        raise OSError("no route to host")

    home_and_block_network(refuse)
    with caplog.at_level(logging.DEBUG, logger="biosuite.core.enrichment"):
        assert en._download_go_obo() is None

    levels = [record.levelno for record in caplog.records]
    assert logging.WARNING in levels, caplog.text
    assert "Could not download the GO ontology" in caplog.text, caplog.text
    assert logging.INFO not in levels or max(levels) >= logging.WARNING, caplog.text


def test_a_cached_ontology_is_used_without_any_transfer(home_and_block_network, tmp_path):
    def explode(*args, **kwargs):
        raise AssertionError("the cache was there and the network was asked anyway")

    home_and_block_network(explode)
    (tmp_path / ".biosuite" / "go-basic.obo").write_text("format-version: 1.2\n",
                                                         encoding="utf-8")

    assert en._download_go_obo().endswith("go-basic.obo")


@pytest.fixture
def home_and_block_network(monkeypatch, tmp_path):
    """Point ``~/.biosuite`` at tmp_path and hand back a setter for `urlopen`.

    The cache directory exists but is empty, so what the helper does next is the
    part under test; nothing here is allowed to reach the network.
    """
    def expanduser(path):
        return str(tmp_path) if path == "~" else path

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    (tmp_path / ".biosuite").mkdir(exist_ok=True)

    def use(replacement):
        monkeypatch.setattr(urllib.request, "urlopen", replacement)

    return use
