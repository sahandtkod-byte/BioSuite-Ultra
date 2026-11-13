"""The Workflow tab: pipeline, batch, GO browser, pathway, GWAS and epitope handlers.

Each handler reads a widget, calls the ``biosuite.core`` module behind the tab and
writes what that module returns into a results box, so every expectation here is
recomputed by calling the same core function the tab wraps — the tab is measured
against its module instead of against a remembered string.  The two plotting
handlers are checked on the figure they hand to the app, and the file handlers are
checked on the message they show when the file is wrong.
"""
import re

import pandas as pd
import pytest

from biosuite.core import sequence
from biosuite.core.epitope import (
    format_epitope_report,
    predict_b_cell_epitopes,
    predict_t_cell_epitopes,
)
from biosuite.core.go_browser import GOBrowser, format_go_results
from biosuite.core.gwas import detect_lead_snps, format_gwas_report, run_gwas
from biosuite.core.pathway_viz import (
    create_custom_pathway,
    create_kegg_style_pathway,
    format_pathway_report,
)
from biosuite.core.workflow.batch import BatchProcessor
from biosuite.core.workflow.pipeline import Pipeline

PIPELINE_TEXT = ("reverse:reverse_complement\n"
                 "translate:translate\n"
                 "a line without a colon\n"
                 "badstep:no_such_function_name")

BATCH_TEXT = "sample1\n\nsample2\nsample3\n   \n"

PROTEIN = ("MEEIQIKEQLQSSVRTVSLVILLLILLLIVANNVDCGFHPNLTFLPLPAGIEWDFKVTYNGG"
           "TFDKLSAQVKEETTPKQW")

GENES = ["EGF", "EGFR", "GRB2", "RAS", "RAF"]


def _normalise(text):
    """Strip the elapsed-time numbers every core summary embeds."""
    return re.sub(r"\d+\.\d+s", "N.s", text).strip()


def _lines(text):
    """Normalised lines, sorted: the batch logs its jobs in completion order."""
    return sorted(line.strip() for line in _normalise(text).splitlines() if line.strip())


def _expected_pipeline(text):
    """The same Pipeline the tab builds, run in the test process."""
    p = Pipeline("gui_pipeline")
    for line in text.split("\n"):
        line = line.strip()
        if ":" in line:
            name, func_name = line.split(":", 1)
            func = getattr(sequence, func_name.strip(), None)
            if func:
                p.add_step(name.strip(), func)
    # Both shipped steps take a parameter called ``seq``; a context entry under any
    # other name is dropped by Pipeline.run(), which left every step with no
    # argument and the run failing.
    p.set_context(seq="ATGAAATTTTAA")
    p.run()
    return p


def _expected_batch(samples):
    bp = BatchProcessor("gui_batch")
    bp.add_samples(samples, lambda x: f"processed_{x}")
    bp.run(max_workers=2)
    return bp


@pytest.fixture
def gwas_csv(tmp_path):
    """Nine counted SNPs, one of them genome-wide significant."""
    rows = [dict(chrom=1, pos=i * 1000, snp_id=f"rs{i:03d}",
                 case_alt=10 + i, case_ref=190 - i,
                 ctrl_alt=12 + i, ctrl_ref=188 - i) for i in range(1, 9)]
    rows.append(dict(chrom=1, pos=50000, snp_id="rsHIT",
                     case_alt=180, case_ref=20, ctrl_alt=20, ctrl_ref=180))
    path = tmp_path / "assoc.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ─── Pipeline builder ────────────────────────────────────────────────────────

def test_run_pipeline_shows_the_summary_the_core_produced(ui):
    expected = _expected_pipeline(PIPELINE_TEXT)
    assert len(expected.steps) == 2, "only two of the four lines name a real function"

    ui.set("pipe_text", PIPELINE_TEXT)
    ui.app._run_pipeline()

    text = ui.text("pipe_results")
    assert _normalise(text).startswith(_normalise(expected.summary()))
    assert "Steps: 2" in text
    assert "[X]" not in text, "the shipped steps must run, not merely be listed"
    assert ("\nSkipped 2 line(s) without a usable step:\n"
            "  a line without a colon\n"
            "  badstep:no_such_function_name\n") in text, \
        "the lines the builder could not use must be named, not dropped silently"
    assert ui.messages == []


def test_run_pipeline_without_any_steps_asks_for_them(ui):
    ui.set("pipe_results", "sentinel text")
    ui.clear("pipe_text")

    ui.app._run_pipeline()

    assert ui.messages_of("warning") == ["Enter pipeline steps first."]
    assert ui.text("pipe_results") == "sentinel text\n", "a refused run must not touch the box"


def test_a_pipeline_with_every_step_usable_reports_no_skips(ui):
    ui.set("pipe_text", "reverse:reverse_complement\ntranslate:translate")
    ui.clear("pipe_results")

    ui.app._run_pipeline()

    text = ui.text("pipe_results")
    assert "Steps: 2" in text
    assert "Skipped" not in text, "a clean step list must not be told about skips"
    assert ui.messages == []


def test_run_pipeline_reports_a_core_failure(ui, monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(Pipeline, "run", boom)
    ui.set("pipe_results", "sentinel text")
    ui.set("pipe_text", PIPELINE_TEXT)

    ui.app._run_pipeline()

    assert ui.messages_of("error") == ["pipeline exploded"]
    assert ui.text("pipe_results") == "sentinel text\n"


# ─── Batch processor ─────────────────────────────────────────────────────────

def test_run_batch_processes_every_nonblank_sample_id(ui):
    samples = [line.strip() for line in BATCH_TEXT.split("\n") if line.strip()]
    assert len(BATCH_TEXT.split("\n")) == 6, "the scripted text really carries blank lines"
    assert samples == ["sample1", "sample2", "sample3"]
    expected = _expected_batch(samples)

    ui.set("batch_text", BATCH_TEXT)
    ui.app._run_batch()

    box = ui.text("batch_results")
    assert _lines(box) == _lines(expected.summary()), \
        "the box must show every job and its outcome, as the processor reported them"
    assert "Samples: 3" in box and "Completed: 3" in box and "Failed: 0" in box
    assert ui.messages == []


def test_run_batch_without_sample_ids_asks_for_them(ui):
    ui.set("batch_results", "sentinel text")
    ui.clear("batch_text")

    ui.app._run_batch()

    assert ui.messages_of("warning") == ["Enter sample IDs first."]
    assert ui.text("batch_results") == "sentinel text\n"


def test_run_batch_names_a_sample_id_the_core_rejected(ui):
    try:
        _expected_batch(["duplicate", "duplicate"])
    except ValueError as exc:
        expected = str(exc)
    else:
        raise AssertionError("BatchProcessor accepted a duplicate sample id")
    assert "duplicate" in expected

    ui.clear("batch_results")
    ui.set("batch_text", "duplicate\nduplicate")
    ui.app._run_batch()

    assert ui.messages_of("error") == [expected]
    assert ui.text("batch_results").strip() == "", "a rejected batch must not report results"


# ─── GO browser ──────────────────────────────────────────────────────────────

def test_go_search_lists_the_terms_the_browser_found(ui):
    go = GOBrowser()
    results = go.search("kinase")
    assert len(results) == 2, "the built-in ontology carries two kinase terms"

    ui.set("go_search", "kinase")
    ui.app._go_search()

    assert ui.text("go_results").strip() == format_go_results(results).strip()
    first = results[0]
    detail = ui.text("go_detail")
    assert f"ID: {first.go_id}" in detail
    assert f"Name: {first.name}" in detail
    for parent in go.get_parents(first.go_id):
        assert f"  {parent.go_id} {parent.name}" in detail, "parents must be listed"
    for child in go.get_children(first.go_id):
        assert f"  {child.go_id} {child.name}" in detail, "children must be listed"
    assert ui.messages == []


def test_go_search_with_an_unknown_term_says_so_and_shows_no_detail(ui):
    assert GOBrowser().search("zzznotaterm") == [], "positive control"
    ui.set("go_search", "zzznotaterm")
    ui.set("go_detail", "sentinel detail")

    ui.app._go_search()

    assert ui.text("go_results").strip() == "No GO terms found."
    assert ui.text("go_detail") == "sentinel detail\n"


def test_go_search_with_an_empty_entry_asks_for_a_term(ui):
    ui.set("go_results", "sentinel results")
    ui.set("go_detail", "sentinel detail")
    ui.clear("go_search")

    ui.app._go_search()

    assert ui.messages_of("warning") == ["Enter a search term."]
    assert ui.text("go_results") == "sentinel results\n"
    assert ui.text("go_detail") == "sentinel detail\n"


@pytest.mark.parametrize("namespace", ["BP", "MF"])
def test_go_browse_writes_the_namespace_terms(ui, namespace):
    terms = GOBrowser().get_namespace_terms(namespace)[:30]
    assert terms, f"the built-in ontology has no {namespace} terms"

    ui.app._go_browse(namespace)

    expected = "".join(f"{t.go_id} {t.name}\n" for t in terms)
    assert ui.text("go_results").strip() == expected.strip()
    assert ui.messages == []


# ─── Pathway visualisation ───────────────────────────────────────────────────

def test_draw_pathway_reports_the_genes_and_shows_their_figure(ui):
    expected = format_pathway_report(create_custom_pathway(GENES))

    ui.set("pathway_genes", ", ".join(GENES))
    ui.app._draw_pathway()

    assert ui.text("pathway_info").strip() == expected.strip()
    for gene in GENES:
        assert gene in ui.text("pathway_info"), "every typed gene should be a node"
    assert ui.plots, "the tab never asked for its figure to be displayed"
    title, fig = ui.plots[-1]
    assert title == "Custom Pathway"
    assert ui.app._plot_history[-1]["name"] == f"Custom Pathway ({len(GENES)} genes)"
    assert fig.axes, "the pathway figure has no axes to draw on"
    assert ui.messages == []


def test_draw_pathway_without_genes_asks_for_some(ui):
    ui.set("pathway_info", "sentinel info")
    ui.clear("pathway_genes")

    ui.app._draw_pathway()

    assert ui.messages_of("warning") == ["Enter gene names."]
    assert ui.text("pathway_info") == "sentinel info\n"


@pytest.mark.parametrize("handler,function,prepare,args", [
    ("_go_search", "biosuite.core.go_browser.GOBrowser.search",
     lambda ui: ui.set("go_search", "kinase"), ()),
    ("_go_browse", "biosuite.core.go_browser.GOBrowser.get_namespace_terms",
     lambda ui: None, ("BP",)),
    ("_draw_pathway", "biosuite.core.pathway_viz.create_custom_pathway",
     lambda ui: ui.set("pathway_genes", "EGF,EGFR"), ()),
    ("_kegg_demo", "biosuite.core.pathway_viz.create_kegg_style_pathway",
     lambda ui: None, ()),
])
def test_a_core_failure_is_reported_instead_of_escaping(ui, monkeypatch,
                                                        handler, function, prepare, args):
    """These four handlers called the core with no error boundary; a failure
    would print a traceback into the Tk callback and leave the user guessing."""
    def broken(*args_, **kwargs):
        raise RuntimeError("the core is unwell")

    monkeypatch.setattr(function, broken)
    prepare(ui)
    ui.set("go_results", "KEEP RESULTS")
    ui.set("go_detail", "KEEP DETAIL")
    ui.set("pathway_info", "KEEP INFO")

    getattr(ui.app, handler)(*args)

    assert ui.messages_of("error") == ["the core is unwell"]
    assert ui.text("go_results") == "KEEP RESULTS\n"
    assert ui.text("pathway_info") == "KEEP INFO\n"


def test_kegg_demo_reports_and_draws_the_reference_pathway(ui):
    pm = create_kegg_style_pathway()
    expected = format_pathway_report(pm)

    ui.app._kegg_demo()

    assert ui.text("pathway_info").strip() == expected.strip()
    assert f"Nodes: {len(pm.nodes)}" in ui.text("pathway_info")
    assert ui.plots[-1][0] == "KEGG-Style Pathway"
    assert ui.app._plot_history[-1]["name"] == "KEGG-Style Pathway"
    assert ui.messages == []


# ─── GWAS ────────────────────────────────────────────────────────────────────

def test_gwas_load_analyses_the_csv_it_was_given(ui, gwas_csv):
    ui.open_path = str(gwas_csv)
    frame = pd.read_csv(str(gwas_csv))
    results = run_gwas(frame)
    report = format_gwas_report(results, detect_lead_snps(results))
    assert "rsHIT" in report, "the fixture must contain a genome-wide significant SNP"

    ui.app._gwas_load()

    box = ui.text("gwas_results")
    assert box.strip() == f"Loaded {len(frame)} SNPs\n\n{report}".strip()
    assert ui.status() == f"GWAS: analysed {len(frame)} SNPs"
    assert ui.messages == []


def test_gwas_load_cancelled_keeps_the_previous_results(ui):
    ui.set("gwas_results", "earlier report")
    before = ui.status()
    ui.open_path = ""

    ui.app._gwas_load()

    assert ui.text("gwas_results") == "earlier report\n"
    assert ui.status() == before
    assert ui.messages == []


def test_gwas_load_names_a_csv_it_cannot_read(ui, tmp_path):
    ui.set("gwas_results", "earlier report")
    ui.open_path = str(tmp_path / "absent.csv")

    ui.app._gwas_load()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert errors[0].startswith("Could not load CSV:") and "absent.csv" in errors[0]
    assert ui.text("gwas_results") == "earlier report\n"


def test_gwas_load_reports_a_csv_without_association_columns(ui, tmp_path):
    path = tmp_path / "notes.csv"
    pd.DataFrame({"gene": ["A", "B"], "score": [1, 2]}).to_csv(path, index=False)
    try:
        run_gwas(pd.read_csv(str(path)))
    except ValueError as exc:
        expected = str(exc)
    else:
        raise AssertionError("run_gwas accepted a frame without count columns")

    ui.set("gwas_results", "")
    ui.open_path = str(path)
    ui.app._gwas_load()

    assert ui.messages_of("error") == [expected]
    assert ui.text("gwas_results").strip() == "Loaded 2 SNPs", \
        "the load succeeded, so the box must say so before the analysis failed"


def test_gwas_demo_writes_the_report_for_its_simulated_data(ui):
    from biosuite.core.gwas import generate_gwas_data

    data = generate_gwas_data(n_snps=2000)
    results = run_gwas(data)
    expected = format_gwas_report(results, detect_lead_snps(results))

    ui.app._gwas_demo()

    assert ui.text("gwas_results").strip() == expected.strip()
    assert f"Total SNPs tested: {len(results)}" in ui.text("gwas_results")
    assert ui.messages == []


# ─── Epitope prediction ──────────────────────────────────────────────────────

def test_run_epitope_shows_the_report_for_the_typed_sequence(ui):
    upper = PROTEIN.upper()
    tc = predict_t_cell_epitopes(upper, mhc_type="A0201")
    bc = predict_b_cell_epitopes(upper)
    assert tc and bc, "the fixture protein must yield epitopes"

    ui.set("epi_seq", PROTEIN.lower())
    ui.set("epi_hla", "A0201")
    ui.app._run_epitope()

    assert ui.text("epi_results").strip() == \
        format_epitope_report(tc, bc, "input").strip()
    assert ui.messages == []


def test_the_hla_entry_reaches_the_t_cell_predictor(ui):
    """The tab must hand what was typed to ``mhc_type``, not a fixed allele."""
    a0201 = format_epitope_report(predict_t_cell_epitopes(PROTEIN, mhc_type="A0201"),
                                  predict_b_cell_epitopes(PROTEIN), "input")
    a0101 = format_epitope_report(predict_t_cell_epitopes(PROTEIN, mhc_type="A0101"),
                                  predict_b_cell_epitopes(PROTEIN), "input")
    assert a0201 != a0101, "positive control: the two alleles must score differently"

    ui.set("epi_seq", PROTEIN)
    ui.set("epi_hla", "A0101")
    ui.app._run_epitope()

    assert ui.text("epi_results").strip() == a0101.strip()


def test_an_empty_hla_entry_falls_back_to_the_default_allele(ui):
    expected = format_epitope_report(predict_t_cell_epitopes(PROTEIN, mhc_type="A0201"),
                                     predict_b_cell_epitopes(PROTEIN), "input")

    ui.set("epi_seq", PROTEIN)
    ui.clear("epi_hla")
    ui.app._run_epitope()

    assert ui.text("epi_results").strip() == expected.strip()
    assert ui.messages == []


def test_run_epitope_without_a_sequence_asks_for_one(ui):
    ui.set("epi_results", "sentinel epitopes")
    ui.clear("epi_seq")

    ui.app._run_epitope()

    assert ui.messages_of("warning") == ["Enter a protein sequence first."]
    assert ui.text("epi_results") == "sentinel epitopes\n"


def test_run_epitope_reports_a_core_failure(ui, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no epitope matrix")

    monkeypatch.setattr("biosuite.core.epitope.predict_t_cell_epitopes", boom)
    ui.set("epi_results", "sentinel epitopes")
    ui.set("epi_seq", PROTEIN)

    ui.app._run_epitope()

    assert ui.messages_of("error") == ["no epitope matrix"]
    assert ui.text("epi_results") == "sentinel epitopes\n"
