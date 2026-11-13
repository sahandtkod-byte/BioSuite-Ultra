"""Contracts pinned here: what ``biosuite/core/enrichment.py`` does when its two
optional engines (goatools, gseapy) are present, and what it does when the
network-backed helpers run.  goatools and gseapy are not installed in this
environment, so the tests inject stand-ins that compute the same statistics
(scipy's hypergeometric, statsmodels' Benjamini-Hochberg, gseapy's result
tables) and assert on the numbers that reach the report.

Why it matters: an enrichment report is a biological claim.  A p-value slot
filled with the wrong correction, a term whose FDR was never adjusted, an
argument silently dropped on the way to the engine, or an empty report that
looks like "nothing was enriched" instead of "no engine available" all produce
papers-worthy output that is wrong.  Every expected number below was measured
against an independent computation in the test file itself.


The engine-agnostic text report lives in ``test_enrichment_text_report.py``
and the ``~/.biosuite`` download/cache helpers in
``test_enrichment_ontology_cache.py``.
"""
import contextlib
import importlib.util
import os
import sys
import time
import types
import urllib.request

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from biosuite.core import enrichment as en

BACKGROUND = [f"G{i}" for i in range(1, 21)]

#: gene -> GO terms.  Seven terms: one deep and strong, one deep with no
#: overlap, one shallow-but-significant, one significant term whose depth is
#: None, one significant term the OBO has never heard of, and the root.
ASSOCIATIONS = {
    "GO:0000002": ["G1", "G2", "G3", "G4"],                      # depth 2, strong
    "GO:0007777": ["G1", "G2", "G3"],                            # depth None
    "GO:0010101": ["G1", "G2", "G3", "G4", "G5"],                # not in the OBO
    "GO:0000006": ["G1", "G2", "G3"],                            # depth 1
    "GO:0000001": list(BACKGROUND),                              # depth 1, everything
    "GO:0000004": [f"G{i}" for i in range(13, 21)],              # depth 2, no overlap
    "GO:0000008": ["G5", "G6"],                                  # depth 3, no overlap
}
DEPTH = {"GO:0000002": 2, "GO:0007777": None, "GO:0000006": 1,
         "GO:0000001": 1, "GO:0000004": 2, "GO:0000008": 3}
NAMES = {"GO:0000002": "cell cycle checkpoint", "GO:0007777": "depth unknown",
         "GO:0000006": "single-organism cellular process", "GO:0000001": "biological_process",
         "GO:0000004": "metabolic process", "GO:0000008": "ribosome biogenesis"}
STUDY = ["G1", "G2", "G3", "G4"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _manual_bh(pvals):
    """Benjamini-Hochberg step-up, written out independently of statsmodels."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, pvals[index] * m / (rank + 1))
        adjusted[index] = running
    return adjusted


def _expected_ora_table(study=STUDY, associations=ASSOCIATIONS, population=BACKGROUND):
    """(raw p, BH, Bonferroni, overlap) per term, computed here from scratch."""
    n, m = len(population), len(study)
    rows = {}
    for go_id, genes in associations.items():
        overlap = len(set(genes) & set(study))
        p = float(hypergeom.sf(overlap - 1, n, len(genes), m))
        rows[go_id] = p
    order = sorted(rows, key=lambda g: rows[g])
    ordered = [rows[g] for g in order]
    bh = _manual_bh(ordered)
    for go_id, adj, p in zip(order, bh, ordered):
        rows[go_id] = (p, adj, min(1.0, p * len(order)), len(set(associations[go_id]) & set(study)))
    return rows


class _Term:
    """One node of the fake OBO graph: goatools exposes `.depth` on it."""

    def __init__(self, depth):
        self.depth = depth


class _FakeObodag(dict):
    """The part of GODag run_ora touches: membership and `.depth` lookup."""


class _Record:
    """A goatools GOEA_result: only the attributes the module reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeGoatoolsStudy:
    """Stand-in for GOEnrichmentStudy that runs the real over-representation test.

    The statistics are the ones goatools uses (hypergeometric upper tail,
    Bonferroni, Benjamini-Hochberg through statsmodels), so what is under test
    stays what the module does with them: which correction it adopts, what it
    filters, how it sorts and how many genes it reports.
    """

    #: set by the test factory, read back by the assertions
    last = None

    def __init__(self, population, associations, obodag, methods=None, cutoff=0.05):
        self.population = sorted(population)
        self.associations = associations
        self.obodag = obodag
        self.methods = list(methods) if methods else []
        self.cutoff = cutoff
        self.studied = None
        FakeGoatoolsStudy.last = self

    def run_study(self, study_genes):
        self.studied = set(study_genes)
        table = _expected_ora_table(study_genes, self._associations_by_term(), self.population)
        records = []
        for go_id, (p_raw, p_bh, p_bonf, overlap) in sorted(
                table.items(), key=lambda item: item[1][0]):
            fields = {"GO": go_id, "name": NAMES.get(go_id, go_id), "p_value": p_raw,
                      "p_fdr_bh": p_bh, "p_bonferroni": p_bonf, "study_count": overlap,
                      "study_items": sorted(set(self._term_genes(go_id)) & set(study_genes))}
            for dropped in self.dropped:
                fields.pop(dropped, None)
            records.append(_Record(**fields))
        return records

    def _associations_by_term(self):
        terms = {}
        for gene, gene_terms in self.associations.items():
            for go_id in gene_terms:
                terms.setdefault(go_id, set()).add(gene)
        return {go_id: sorted(genes) for go_id, genes in terms.items()}

    def _term_genes(self, go_id):
        return self._associations_by_term()[go_id]

    dropped = ()


def _goatools_fixture(monkeypatch, obodag=None, dropped=(), error=None):
    """Wire the fake engine into the module under test and return the DAG used."""
    dag = _FakeObodag() if obodag is None else obodag
    if obodag is None:
        for go_id, depth in DEPTH.items():
            dag[go_id] = _Term(depth)

    class _Study(FakeGoatoolsStudy):
        pass

    _Study.dropped = tuple(dropped)
    if error is not None:
        def boom(self, study_genes):
            raise error
        _Study.run_study = boom
    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GODag", lambda path: dag, raising=False)
    monkeypatch.setattr(en, "GOEnrichmentStudy", _Study, raising=False)
    return dag


def _association_file(tmp_path):
    lines = [f"{gene}\t{go_id}" for go_id, genes in ASSOCIATIONS.items() for gene in genes]
    path = tmp_path / "associations.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _obo_file(tmp_path):
    """A go-basic style OBO whose only real job here is to exist: GODag is faked."""
    path = tmp_path / "go-basic.obo"
    path.write_text("format-version: 1.2\n", encoding="utf-8")
    return str(path)


class _FakeEngineResult:
    def __init__(self, frame):
        self.res2d = frame


class FakeGseapy:
    """Records the calls gseapy would receive and answers with a result table."""

    def __init__(self, prerank_result=None, enrich_result=None, error=None):
        self.prerank_result = prerank_result
        self.enrich_result = enrich_result
        self.error = error
        self.prerank_calls = []
        self.enrich_calls = []

    def prerank(self, **kwargs):
        self.prerank_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.prerank_result

    def enrich(self, **kwargs):
        self.enrich_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.enrich_result


@contextlib.contextmanager
def _executed_with(**modules):
    """Run enrichment.py again with ``modules`` (import name -> module) injected.

    ``HAS_GOATOOLS``/``HAS_GSEAPY`` and the statsmodels compatibility shim are
    decided while the module body runs, so the only way to reach the "engine
    present" side of those guards is to execute the file with the dependency
    substituted.  Entries are parked and restored, leaving the imported module
    and the real third-party packages untouched.
    """
    suffix = "_".join(sorted(name.replace(".", "_") for name in modules))
    alias = f"biosuite.core._enrichment_with_{suffix}"
    parked = {}
    for name in list(modules) + ["statsmodels.sandbox.stats",
                                 "statsmodels.sandbox.stats.multicomp"]:
        if name in sys.modules:
            parked[name] = sys.modules[name]
    parent_attributes = []
    sys.modules.update(modules)
    for name, module in modules.items():
        if "." in name:
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                parent_attributes.append((parent, child, getattr(parent, child, None)))
                setattr(parent, child, module)
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
        for parent, child, value in parent_attributes:
            setattr(parent, child, value)


@pytest.fixture
def home_free(monkeypatch, tmp_path):
    """Point every ``~/.biosuite`` helper at tmp_path instead of the real home."""
    def expanduser(path):
        return str(tmp_path) if path == "~" else path
    monkeypatch.setattr(os.path, "expanduser", expanduser)
    return tmp_path


# ── engine detection and the report container ────────────────────────────────

def test_check_enrichment_tools_answers_for_both_optional_engines():
    tools = en.check_enrichment_tools()
    assert sorted(tools) == ["goatools", "gseapy"]
    assert all(isinstance(value, bool) for value in tools.values())
    assert tools["goatools"] is en.HAS_GOATOOLS
    assert tools["gseapy"] is en.HAS_GSEAPY


def test_the_engine_field_names_the_backend_that_can_actually_run_the_analysis():
    assert en._engine_for("ORA") == ("goatools" if en.HAS_GOATOOLS else "none")
    assert en._engine_for("GSEA") == ("gseapy" if en.HAS_GSEAPY else "none")
    assert en._engine_for("KEGG") == ("gseapy" if en.HAS_GSEAPY else "none")


def test_a_report_keeps_the_engine_the_caller_named_explicitly():
    report = en.EnrichmentReport("ORA", 12, 3, engine="manual")
    assert report.engine == "manual"
    assert en.EnrichmentReport("GSEA", 12, 3).engine == en._engine_for("GSEA")


def test_top_terms_and_significant_terms_slice_the_same_result_list():
    strong = en.EnrichmentResult("ORA", "apoptosis", "GO:0006915", 1e-6, 0.001)
    weak = en.EnrichmentResult("ORA", "digestion", "GO:0052727", 0.2, 0.6)
    report = en.EnrichmentReport("ORA", 20, 1, [strong, weak])
    assert report.top_terms(1) == [strong]
    assert report.top_terms(5) == [strong, weak]
    assert report.significant_terms(0.05) == [strong]
    assert report.significant_terms(0.0005) == []
    assert report.significant_terms()[0].term_id == "GO:0006915"


def test_the_import_guards_take_the_engine_present_side_when_the_packages_exist():
    """Lines 27-29, 32-33 and 39: the shim plus both HAS_* flags."""
    sandbox = types.ModuleType("statsmodels.sandbox.stats.multicomp")
    goatools = types.ModuleType("goatools")
    goatools.__path__ = []
    go_enrichment = types.ModuleType("goatools.go_enrichment")
    go_enrichment.GOEnrichmentStudy = object
    obo_parser = types.ModuleType("goatools.obo_parser")
    obo_parser.GODag = object
    gseapy = types.ModuleType("gseapy")
    gseapy.prerank = lambda **kwargs: None
    with _executed_with(
            **{"statsmodels.sandbox.stats.multicomp": sandbox,
               "goatools": goatools,
               "goatools.go_enrichment": go_enrichment,
               "goatools.obo_parser": obo_parser,
               "gseapy": gseapy}) as module:
        assert module.HAS_GOATOOLS is True and module.HAS_GSEAPY is True
        assert sandbox.multipletests is multipletests, \
            "the relocated multipletests symbol was not restored for goatools"
        assert module._engine_for("ORA") == "goatools"
        assert module._engine_for("GSEA") == "gseapy"


def test_the_shim_is_left_alone_when_the_legacy_module_still_exports_the_symbol():
    """statsmodels >= 0.15 still ships multipletests in the sandbox path.

    Only when it does not should the shim reach in and re-export the relocated
    symbol, so a working legacy module must come out of the import untouched.
    """
    sandbox = types.ModuleType("statsmodels.sandbox.stats.multicomp")
    sentinel = lambda *args, **kwargs: "legacy implementation"  # noqa: E731
    sandbox.multipletests = sentinel
    goatools = types.ModuleType("goatools")
    goatools.__path__ = []
    go_enrichment = types.ModuleType("goatools.go_enrichment")
    go_enrichment.GOEnrichmentStudy = object
    obo_parser = types.ModuleType("goatools.obo_parser")
    obo_parser.GODag = object
    with _executed_with(**{"statsmodels.sandbox.stats.multicomp": sandbox,
                           "goatools": goatools,
                           "goatools.go_enrichment": go_enrichment,
                           "goatools.obo_parser": obo_parser}) as module:
        assert module.HAS_GOATOOLS is True
        assert module.HAS_GSEAPY is en.HAS_GSEAPY
        assert sandbox.multipletests is sentinel, "a working legacy symbol was overwritten"


def test_the_session_module_is_not_rewritten_by_the_injected_imports():
    """The reason a fresh module object is used instead of patching the real one."""
    assert hasattr(en, "GOEnrichmentStudy") is en.HAS_GOATOOLS
    assert hasattr(en, "GODag") is en.HAS_GOATOOLS
    assert hasattr(en, "gp") is en.HAS_GSEAPY
    assert en._engine_for("ORA") == ("goatools" if en.HAS_GOATOOLS else "none")


# ── ORA: the goatools path ───────────────────────────────────────────────────

def test_ora_answers_an_empty_gene_list_before_touching_the_engine(monkeypatch):
    _goatools_fixture(monkeypatch)
    report = en.run_ora([])
    assert report.analysis_type == "ORA"
    assert (report.num_input_genes, report.num_significant, report.results) == (0, 0, [])
    assert report.message == "Empty gene list provided."


def test_ora_fetches_the_obo_and_the_organism_associations_when_neither_is_given(
        monkeypatch, tmp_path):
    _goatools_fixture(monkeypatch)
    calls = []

    def fake_obo():
        calls.append("obo")
        return _obo_file(tmp_path)

    def fake_associations(organism):
        calls.append(f"associations:{organism}")
        return {}

    monkeypatch.setattr(en, "_download_go_obo", fake_obo)
    monkeypatch.setattr(en, "_get_associations", fake_associations)
    report = en.run_ora(STUDY, organism="mouse")
    assert calls == ["obo", "associations:mouse"]
    assert report.num_significant == 0
    assert report.message == ("Could not load gene associations: no gene/GO-term "
                              "associations are available for organism 'mouse'.")


def test_ora_accepts_organism_associations_in_place_of_an_association_file(
        monkeypatch, tmp_path):
    """The two association sources must feed the study identically."""
    from_file = _goatools_fixture(monkeypatch)
    gene_to_terms = {gene: set() for gene in BACKGROUND}
    for go_id, genes in ASSOCIATIONS.items():
        for gene in genes:
            gene_to_terms[gene].add(go_id)
    monkeypatch.setattr(en, "_get_associations", lambda organism: dict(gene_to_terms))
    monkeypatch.setattr(en, "_download_go_obo", lambda: _obo_file(tmp_path))
    from_database = en.run_ora(STUDY, organism="human")
    from_disk = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                           associations_file=_association_file(tmp_path))
    assert from_database.results == from_disk.results
    assert from_database.background_count == from_disk.background_count == 20
    assert [r.term_id for r in from_database.results] == ["GO:0000002", "GO:0010101"]
    assert from_file["GO:0000002"].depth == 2


def test_ora_reports_the_id_space_when_no_input_gene_matches(monkeypatch, tmp_path):
    """"No input genes found in association database." told the user nothing.

    The background is keyed by Entrez gene IDs, so a symbol list can never match;
    the message now says how many identifiers were asked, how big the background
    is and which key space it uses — and the background count survives the early
    return, which it did not.
    """
    _goatools_fixture(monkeypatch)
    report = en.run_ora(["NO_SUCH_GENE"], obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    assert report.num_input_genes == 1
    assert report.num_significant == 0
    assert report.background_count == 20, "the background was loaded and is reportable"
    assert report.message == (
        "None of the 1 input identifier(s) is in the 20-gene background. That "
        "background is keyed by Entrez gene IDs (numbers as text), so a symbol "
        "such as 'NO_SUCH_GENE' can never match."), report.message
    assert report.results == []


def test_ora_keeps_significant_deep_terms_and_sorts_them_by_the_reported_correction(
        monkeypatch, tmp_path):
    """Two of the four significant terms are dropped for GO depth, and the report says so."""
    _goatools_fixture(monkeypatch)
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path), method="fdr_bh")
    expected = _expected_ora_table()
    assert [r.term_id for r in report.results] == ["GO:0000002", "GO:0010101"]
    assert report.results[0].adjusted_p_value == pytest.approx(expected["GO:0000002"][1])
    assert report.results[0].p_value == pytest.approx(expected["GO:0000002"][0])
    assert report.results[1].adjusted_p_value == pytest.approx(expected["GO:0010101"][1])
    assert report.num_significant == 2
    assert report.num_input_genes == 4
    assert report.background_count == 20
    assert report.engine == "goatools"
    assert report.correction == "FDR"
    # The default method is Bonferroni, so the same run reports that number instead.
    bonferroni = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                            associations_file=_association_file(tmp_path))
    assert bonferroni.correction == "Bonferroni"
    assert [r.adjusted_p_value for r in bonferroni.results] == [
        pytest.approx(expected["GO:0000002"][2]), pytest.approx(expected["GO:0010101"][2])]
    assert bonferroni.message == ("2 term(s) cleared the cutoff but were dropped "
                                  "for a GO depth below 2 (too general to act on)")


def test_ora_hypergeometric_and_bh_are_what_an_independent_computation_predicts(
        monkeypatch, tmp_path):
    """The cross-check: 4 of 4 study genes in a term of size 4 out of 20."""
    _goatools_fixture(monkeypatch)
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    best = report.results[0]
    raw = float(hypergeom.sf(3, 20, 4, 4))
    assert raw == pytest.approx(0.0002063983488132095)
    assert best.p_value == pytest.approx(raw)
    assert best.p_value < best.adjusted_p_value <= 1.0
    assert best.gene_count == 4
    assert sorted(best.genes) == ["G1", "G2", "G3", "G4"]
    assert best.term_name == "cell cycle checkpoint"
    assert best.description == "cell cycle checkpoint"
    assert best.category == "BP"
    assert best.analysis_type == "ORA"


def test_ora_reports_the_nominal_p_alongside_the_correction_not_a_second_correction(
        monkeypatch, tmp_path):
    """The regression: p_value was filled with r.p_bonferroni.

    Bonferroni over seven tests is roughly seven times the nominal p, so the
    field named p_value disagreed with the raw hypergeometric by that factor
    and the uncorrected number was nowhere in the report - unlike run_gsea,
    which reads gseapy's nominal 'NOM p-val' into the same slot.
    """
    _goatools_fixture(monkeypatch)
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    best = report.results[0]
    expected = _expected_ora_table()["GO:0000002"]
    assert best.p_value == pytest.approx(expected[0])
    assert best.p_value != pytest.approx(expected[2])
    assert best.adjusted_p_value == pytest.approx(expected[1])


def test_ora_asks_for_both_corrections_whichever_method_the_caller_named(monkeypatch, tmp_path):
    dag = _goatools_fixture(monkeypatch)
    en.run_ora(STUDY, obo_file=_obo_file(tmp_path), associations_file=_association_file(tmp_path))
    assert FakeGoatoolsStudy.last.methods == ["bonferroni", "fdr_bh"]
    assert FakeGoatoolsStudy.last.cutoff == 0.05
    assert FakeGoatoolsStudy.last.studied == set(STUDY)
    assert sorted(FakeGoatoolsStudy.last.population) == sorted(BACKGROUND)
    en.run_ora(STUDY, obo_file=_obo_file(tmp_path), associations_file=_association_file(tmp_path),
               method="sidak", cutoff=0.001)
    assert FakeGoatoolsStudy.last.methods == ["bonferroni", "fdr_bh", "sidak"]
    assert FakeGoatoolsStudy.last.cutoff == 0.001
    assert dag["GO:0000002"].depth == 2


def test_ora_uses_bonferroni_when_the_record_carries_no_bh_value(monkeypatch, tmp_path):
    """p_fdr_bh is read first; without it the Bonferroni number becomes the FDR.

    For the most significant term the two coincide (BH divides by rank 1), so
    the second term is the one that shows the substitution.
    """
    _goatools_fixture(monkeypatch, dropped=("p_fdr_bh",))
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    expected = _expected_ora_table()
    assert [r.term_id for r in report.results] == ["GO:0000002", "GO:0010101"]
    assert report.results[0].adjusted_p_value == pytest.approx(expected["GO:0000002"][2])
    assert report.results[0].p_value == pytest.approx(expected["GO:0000002"][0])
    assert report.results[1].adjusted_p_value == pytest.approx(expected["GO:0010101"][2])
    assert report.results[1].adjusted_p_value > expected["GO:0010101"][1]


def test_ora_skips_a_record_that_carries_neither_correction(monkeypatch, tmp_path):
    _goatools_fixture(monkeypatch, dropped=("p_fdr_bh", "p_bonferroni"))
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    assert report.results == []
    assert report.num_significant == 0
    assert report.background_count == 20


def test_ora_drops_a_shallow_term_and_a_term_of_unknown_depth(monkeypatch, tmp_path):
    """GO:0000006 is significant at depth 1, GO:0007777 significant with depth None."""
    _goatools_fixture(monkeypatch)
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    kept = {r.term_id for r in report.results}
    expected = _expected_ora_table()
    assert kept == {"GO:0000002", "GO:0010101"}
    assert expected["GO:0000006"][1] < 0.05, "the shallow term had to clear the cutoff"
    assert expected["GO:0007777"][1] < 0.05, "the depthless term had to clear the cutoff"


def test_ora_applies_no_depth_filter_to_terms_the_obo_does_not_know(monkeypatch, tmp_path):
    """Membership is checked before depth, so an empty DAG filters nothing.

    With the fixture's DAG the shallow term and the term of unknown depth are
    dropped; with no DAG at all all four significant terms survive, which is how
    a truncated go-basic.obo would behave in production.

    GO:0000006 and GO:0007777 come out with the same raw p, the same Bonferroni
    and the same BH value, so their order is a real tie.  The report breaks it on
    the term id: the sequence the engine hands over comes from a set traversal,
    and without that tie-break PYTHONHASHSEED decided the order, so one gene
    list produced a differently-ordered report in each run.
    """
    _goatools_fixture(monkeypatch, obodag=_FakeObodag())
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    assert [r.term_id for r in report.results] == ["GO:0000002", "GO:0010101",
                                                  "GO:0000006", "GO:0007777"]
    tied = {r.term_id: r.adjusted_p_value for r in report.results
            if r.term_id in ("GO:0000006", "GO:0007777")}
    assert tied["GO:0000006"] == tied["GO:0007777"], "the tie is real"
    assert report.num_significant == 4


def test_ora_tolerates_a_record_without_a_name_or_a_gene_list(monkeypatch, tmp_path):
    _goatools_fixture(monkeypatch, dropped=("name", "study_items"))
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    best = report.results[0]
    assert best.term_name == "" and best.description == ""
    assert best.term_id == "GO:0000002"
    assert best.genes == []
    assert best.gene_count == 4


def test_ora_surfaces_an_engine_failure_as_a_message_not_an_exception(monkeypatch, tmp_path):
    _goatools_fixture(monkeypatch, error=RuntimeError("gene2go went missing"))
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=_association_file(tmp_path))
    assert report.num_significant == 0
    assert report.results == []
    assert report.message == "ORA analysis error: gene2go went missing"
    assert report.num_input_genes == 4


def test_ora_reports_an_unreadable_association_file_as_no_associations(monkeypatch, tmp_path):
    _goatools_fixture(monkeypatch)
    report = en.run_ora(STUDY, obo_file=_obo_file(tmp_path),
                        associations_file=str(tmp_path / "absent.tsv"))
    assert report.num_significant == 0
    assert "Could not load gene associations" in report.message


# ── GSEA: the gseapy prerank path ────────────────────────────────────────────

def _gsea_fixture(monkeypatch, frame=None, result=None, error=None):
    engine = FakeGseapy(prerank_result=_FakeEngineResult(frame) if result is None else result,
                        error=error)
    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    monkeypatch.setattr(en, "gp", engine, raising=False)
    return engine


GSEA_ROWS = {
    "Term": ["SetA apoptosis", "SetB ribosome"],
    "NES": [1.85, -1.42],
    "NOM p-val": [0.001, np.nan],
    "FDR q-val": [0.02, 0.4],
    "Genes": ["G1;G2;G3", np.nan],
}


def test_gsea_answers_an_empty_gene_list_before_ranking_anything(monkeypatch):
    _gsea_fixture(monkeypatch, frame=pd.DataFrame(GSEA_ROWS))
    report = en.run_gsea([])
    assert (report.num_input_genes, report.num_significant, report.results) == (0, 0, [])
    assert report.message == "Empty gene list provided."
    assert report.analysis_type == "GSEA"


def test_gsea_ranks_by_the_supplied_scores_from_high_to_low(monkeypatch):
    engine = _gsea_fixture(monkeypatch, frame=pd.DataFrame(GSEA_ROWS))
    report = en.run_gsea(["G1", "G2", "G3"], gene_scores=[1.5, -4.0, 2.25],
                         gene_sets="my_sets.gmt", min_size=3, max_size=40,
                         permutation_num=25, seed=7)
    ranking = engine.prerank_calls[0]["rnk"]
    assert list(ranking.index) == ["G3", "G1", "G2"]
    assert list(ranking.values) == [2.25, 1.5, -4.0]
    assert engine.prerank_calls[0]["gene_sets"] == "my_sets.gmt"
    assert (engine.prerank_calls[0]["min_size"], engine.prerank_calls[0]["max_size"]) == (3, 40)
    assert engine.prerank_calls[0]["permutation_num"] == 25
    assert engine.prerank_calls[0]["no_plot"] is True and engine.prerank_calls[0]["silent"] is True
    assert report.analysis_type == "GSEA"


def test_gsea_falls_back_to_list_position_when_no_scores_are_supplied(monkeypatch):
    engine = _gsea_fixture(monkeypatch, frame=pd.DataFrame(GSEA_ROWS))
    en.run_gsea(["G1", "G2", "G3"], organism="Mouse", permutation_num=5)
    ranking = engine.prerank_calls[0]["rnk"]
    assert list(ranking.index) == ["G1", "G2", "G3"]
    assert list(ranking.values) == [3, 2, 1]
    assert engine.prerank_calls[0]["organism"] == "Mouse"


def test_gsea_turns_the_result_table_into_sorted_terms(monkeypatch):
    _gsea_fixture(monkeypatch, frame=pd.DataFrame(GSEA_ROWS))
    report = en.run_gsea(BACKGROUND, gene_scores=list(np.linspace(2, -2, 20)))
    assert [r.term_name for r in report.results] == ["SetA apoptosis", "SetB ribosome"]
    a, b = report.results
    assert a.p_value == pytest.approx(0.001)
    assert a.adjusted_p_value == pytest.approx(0.02)
    assert a.enrichment_score == pytest.approx(1.85)
    assert a.genes == ["G1", "G2", "G3"] and a.gene_count == 3
    assert a.term_id == "SetA apoptosis"
    assert b.p_value != b.p_value, "a missing nominal p stays missing, it is not 1.0"
    assert b.adjusted_p_value == pytest.approx(0.4)
    assert b.genes == [] and b.gene_count == 0
    assert report.num_significant == 1
    assert report.engine == "gseapy"
    assert report.correction == "FDR"
    assert "no 'NOM p-val'" in report.message, report.message


def test_gsea_sorts_by_the_adjusted_p_value_not_by_table_order(monkeypatch):
    frame = pd.DataFrame({"Term": ["first", "second", "third"],
                          "NES": [1.1, 2.2, -3.3],
                          "NOM p-val": [0.4, 0.001, 0.02],
                          "FDR q-val": [0.9, 0.03, 0.04],
                          "Genes": ["G1", "G2;G3", "G4"]})
    _gsea_fixture(monkeypatch, frame=frame)
    report = en.run_gsea(BACKGROUND)
    assert [r.term_name for r in report.results] == ["second", "third", "first"]
    assert [r.adjusted_p_value for r in report.results] == pytest.approx([0.03, 0.04, 0.9])


def test_gsea_reports_when_the_engine_returns_no_table(monkeypatch):
    engine = _gsea_fixture(monkeypatch, result=_FakeEngineResult(None))
    report = en.run_gsea(["G1", "G2"], gene_scores=[1.0, 2.0])
    assert report.message == "GSEA returned no results."
    assert report.num_input_genes == 2
    assert report.results == []
    assert len(engine.prerank_calls) == 1


def test_gsea_reports_an_engine_failure_without_raising(monkeypatch):
    _gsea_fixture(monkeypatch, error=ValueError("gmt file unreadable"))
    report = en.run_gsea(["G1", "G2"])
    assert report.message == "GSEA error: gmt file unreadable"
    assert report.num_significant == 0


def test_gsea_without_the_engine_names_the_engine_and_why_it_is_unusable(monkeypatch):
    """The other half of N18: an unavailable backend is not an empty result.

    Returning before the engine is touched is what makes this distinguishable,
    so ``gp`` is parked on ``None`` — any attempt to call it would raise.  The old
    text claimed "not installed" whatever the import failure had been, including a
    broken dependency of a package that was installed.
    """
    monkeypatch.setattr(en, "HAS_GSEAPY", False)
    monkeypatch.setattr(en, "gp", None, raising=False)

    report = en.run_gsea(["G1", "G2", "G3"])

    assert report.engine == "none"
    assert report.message.startswith("gseapy is not usable. Run: pip install gseapy"), \
        report.message
    assert (report.num_input_genes, report.num_significant, report.results) == (3, 0, [])


# ── KEGG: the gseapy enrich path ─────────────────────────────────────────────

KEGG_ROWS = {
    "Term": ["hsa04110 Cell cycle", ""],
    "P-value": [0.0002, 0.3],
    "Adjusted P-value": [0.004, 0.6],
    "Overlap": ["4/190", "0/50"],
    "Description": ["", "KEGG_PATHWAY hsa04211"],
}


def test_kegg_says_which_engine_is_unavailable_when_gseapy_is_missing():
    if en.HAS_GSEAPY:
        pytest.skip("gseapy is installed, so the missing-engine guard cannot fire")
    report = en.run_kegg_enrichment(["P53"])
    assert report.analysis_type == "KEGG"
    assert report.num_significant == 0 and report.results == []
    assert report.message.startswith("gseapy is not usable. Run: pip install gseapy"), \
        report.message


_NO_RESULT = object()


def _kegg_fixture(monkeypatch, frame=None, result=_NO_RESULT, error=None):
    engine = FakeGseapy(
        enrich_result=(_FakeEngineResult(frame) if result is _NO_RESULT else result),
        error=error)
    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    monkeypatch.setattr(en, "gp", engine, raising=False)
    return engine


def test_kegg_defaults_to_the_organism_library_and_the_documented_size_limits(monkeypatch):
    engine = _kegg_fixture(monkeypatch, frame=pd.DataFrame(KEGG_ROWS))
    report = en.run_kegg_enrichment(["6714"], organism="hsa")
    call = engine.enrich_calls[0]
    assert call["gene_sets"] == "KEGG_HSA_2021"
    assert (call["min_size"], call["max_size"]) == (10, 500)
    assert call["organism"] == "hsa" and call["no_plot"] is True
    assert report.analysis_type == "KEGG"


def test_kegg_forwards_a_custom_gene_set_collection_to_the_engine(monkeypatch):
    """The regression: gene_sets, min_size and max_size were never forwarded.

    A caller passing its own GMT was analysed against the built-in KEGG library
    anyway, with gseapy's own size limits rather than the requested ones.
    """
    custom = {"pathway a": ["G1", "G2"]}
    engine = _kegg_fixture(monkeypatch, frame=pd.DataFrame(KEGG_ROWS))
    en.run_kegg_enrichment(["G1"], organism="mmu", gene_sets=custom, min_size=5, max_size=200)
    call = engine.enrich_calls[0]
    assert call["gene_sets"] == custom
    assert (call["min_size"], call["max_size"]) == (5, 200)
    assert call["organism"] == "mmu"


def test_kegg_reads_term_overlap_and_counts_significant_pathways(monkeypatch):
    _kegg_fixture(monkeypatch, frame=pd.DataFrame(KEGG_ROWS))
    report = en.run_kegg_enrichment(["6714"])
    best, second = report.results
    assert best.term_name == "hsa04110 Cell cycle"
    assert best.term_id == "hsa04110 Cell cycle"
    assert best.p_value == pytest.approx(0.0002)
    assert best.adjusted_p_value == pytest.approx(0.004)
    assert best.gene_count == 4
    assert best.category == "KEGG"
    assert second.gene_count == 0, "an overlap of 0/50 still counts as zero genes"
    assert report.num_significant == 1


def test_kegg_falls_back_to_the_description_column_when_the_term_is_missing(monkeypatch):
    frame = pd.DataFrame({"Description": ["KEGG_PATHWAY hsa00010"], "P-value": [0.01],
                          "Adjusted P-value": [0.04], "Overlap": ["3/12"]})
    _kegg_fixture(monkeypatch, frame=frame)
    report = en.run_kegg_enrichment(["G1"])
    assert report.results[0].term_name == "KEGG_PATHWAY hsa00010"
    assert report.results[0].term_id == ""
    assert report.results[0].gene_count == 3
    assert report.num_significant == 1


def test_kegg_keeps_the_other_pathways_when_one_overlap_string_is_not_a_fraction(
        monkeypatch):
    """One unreadable cell is a bad row, not a lost analysis.

    The '/' guard never checked the numerator, so 'n/a' raised inside the loop and
    the whole run came back as an error report with every pathway gone.  The row is
    kept with a gene count of 0 and the count of unreadable cells is said out loud.
    """
    frame = pd.DataFrame({"Term": ["hsa00010", "hsa04110"], "P-value": [0.01, 0.02],
                          "Adjusted P-value": [0.04, 0.03], "Overlap": ["n/a", "5/9"]})
    _kegg_fixture(monkeypatch, frame=frame)
    report = en.run_kegg_enrichment(["G1"])

    assert [r.term_id for r in report.results] == ["hsa04110", "hsa00010"]
    by_term = {r.term_id: r for r in report.results}
    assert by_term["hsa00010"].gene_count == 0, "the unreadable cell is not a number"
    assert by_term["hsa04110"].gene_count == 5
    assert report.num_significant == 2, "both pathways are still significant"
    assert report.message == ("1 pathway(s) had an 'Overlap' cell the module could "
                              "not read; their gene count is reported as 0"), report.message


def test_kegg_accepts_a_plain_dataframe_instead_of_a_result_object(monkeypatch):
    frame = pd.DataFrame({"Term": ["hsa00010 Glycolysis"], "P-value": [0.001],
                          "Adjusted P-value": [0.01], "Overlap": ["12/300"]})
    _kegg_fixture(monkeypatch, result=frame)
    report = en.run_kegg_enrichment(["G1"])
    assert report.results[0].gene_count == 12
    assert report.num_significant == 1


def test_kegg_reports_when_the_engine_returns_nothing(monkeypatch):
    _kegg_fixture(monkeypatch, result=None)
    report = en.run_kegg_enrichment(["G1"])
    assert report.message == "KEGG enrichment returned no results."
    assert report.results == []


def test_kegg_reports_an_engine_failure_without_raising(monkeypatch):
    _kegg_fixture(monkeypatch, error=KeyError("Adjusted P-value"))
    report = en.run_kegg_enrichment(["G1", "G2"])
    assert report.num_input_genes == 2
    assert report.num_significant == 0
    assert "'Adjusted P-value'" in report.message
