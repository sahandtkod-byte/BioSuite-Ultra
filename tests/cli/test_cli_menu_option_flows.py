"""The multi-step menu flows: pipeline, batch, provenance, plugins, interactive plots.

The two sweeps drive every option with generic answers, so the options that ask a
*sequence* of related questions — build a pipeline, choose a provenance action, pick a
plot type — only ever ran their opening lines.  This file scripts those sequences.  Two
of the flows it covers were broken in a way no generic answer could show: the Pipeline
Builder never asked each step for its arguments, so every step of every pipeline built
in the CLI died with ``gc_content() missing 1 required positional argument: 'seq'``,
and the Batch Processor called its input "Sample IDs" while passing each one straight
to the analysis function, so naming two samples failed on both.
"""
import builtins
import io
import json
import re
import socket
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from biosuite.cli import menu
from biosuite.plotting import biological_plots

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DNA = "ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTCGGG"

INTERNAL_ERRORS = (NameError, AttributeError, TypeError, IndexError, UnboundLocalError,
                   ZeroDivisionError, KeyError)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is not allowed in the menu flow tests")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """Answer a scripted sequence of prompts once, and report what the option said.

    The small data set in here exists because the options under test read files by the
    name the user types, in the working directory.
    """
    (tmp_path / "regions.bed").write_text("chr1\t100\t200\tgeneA\nchr1\t300\t400\tgeneB\n",
                                          encoding="utf-8")
    (tmp_path / "points.csv").write_text("x,y\n1,2\n2,4\n3,7\n4,8\n", encoding="utf-8")
    (tmp_path / "labels.csv").write_text("a,b\ncat,dog\ncat,bird\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(menu.config, "quiet", True)

    def run(choice, answers):
        iterator = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(iterator, ""))
        figures_before = set(plt.get_fignums())
        broken = None
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                menu._handle_menu_choice(choice)
            except INTERNAL_ERRORS as exc:
                broken = exc
        out = ANSI.sub("", buffer.getvalue())
        written = sorted(p.name for p in Path.cwd().rglob("*") if p.is_file())
        assert broken is None, \
            f"option {choice} raised {type(broken).__name__}: {broken}"
        return {"out": out, "files": written,
                "leaked": set(plt.get_fignums()) - figures_before}

    return run


# ── Pipeline Builder (92) ────────────────────────────────────────────────────

def test_pipeline_steps_run_with_their_arguments(flow):
    """The builder has to ask what each step receives — a step with no arguments
    cannot call anything that analyses a sequence.

    ``FAILED`` is the measured before-image of this test: every step died with
    ``gc_content() missing 1 required positional argument: 'seq'``.
    """
    result = flow("92", ["first=gc_content", "ATCGATCG", "second=length", "ATCGATCG", ""])

    assert "Running: first" in result["out"] and "Running: second" in result["out"]
    assert "FAILED" not in result["out"], result["out"]
    assert "finished" in result["out"]


def test_pipeline_reports_a_line_that_is_not_a_step(flow):
    result = flow("92", ["just-a-word", ""])

    assert "Not a step: 'just-a-word'" in result["out"]
    assert "No steps added" in result["out"]


def test_pipeline_refuses_a_function_outside_the_package(flow):
    result = flow("92", ["bad=os.system", "", ""])

    assert "Unknown function" in result["out"]
    assert "No steps added" in result["out"]


# ── Batch Processor (93) ─────────────────────────────────────────────────────

def test_batch_runs_the_function_on_each_value(flow):
    result = flow("93", ["gc_content", "ATCG,GGCC"])

    assert "FAILED" not in result["out"], result["out"]
    assert "ATCG" in result["out"] and "GGCC" in result["out"]


def test_batch_with_no_values_says_so(flow):
    result = flow("93", ["gc_content", ""])

    assert "No values given" in result["out"]


def test_batch_with_an_unknown_function_stops(flow):
    result = flow("93", ["shutil:rmtree"])

    assert "Unknown function" in result["out"] or "Only functions" in result["out"]


# ── Provenance Tracker (e) ───────────────────────────────────────────────────

@pytest.mark.parametrize(("action", "answers", "expected"), [
    ("1", ["flow.db"], "Session started:"),
    ("3", ["flow.db"], "Provenance Summary"),
])
def test_provenance_session_and_history(flow, action, answers, expected):
    result = flow("e", [action] + answers)

    assert expected in result["out"]


def test_provenance_records_a_step_and_exports_it(flow, tmp_path):
    recorded = flow("e", ["2", "flow.db", "sequence", "gc_content", '{"sequence": "ATCG"}',
                          "50% GC"])
    assert "Step recorded." in recorded["out"]

    html = flow("e", ["4", "flow.db", ""])
    assert "Saved:" in html["out"]
    assert "provenance.html" in html["files"]

    exported = flow("e", ["5", "flow.db", ""])
    assert "Saved:" in exported["out"]
    assert "provenance.json" in exported["files"]
    body = (tmp_path / "provenance.json").read_text(encoding="utf-8")
    assert json.loads(body), "the JSON export came out empty"
    assert "gc_content" in body


def test_provenance_keeps_parameters_that_are_not_json(flow, tmp_path):
    """The field is free text; a non-JSON answer must be stored, not lost."""
    result = flow("e", ["2", "flow.db", "sequence", "gc_content", "not json at all", "note"])
    assert "Step recorded." in result["out"]

    flow("e", ["5", "flow.db", "flow.json"])
    body = (tmp_path / "flow.json").read_text(encoding="utf-8")
    assert "not json at all" in body


# ── Interactive plots (d) ────────────────────────────────────────────────────

@pytest.mark.parametrize("plot_type", ["1", "2", "3", "4", "5", "6"])
def test_every_interactive_plot_type_writes_html(flow, plot_type, tmp_path):
    result = flow("d", [plot_type, f"plot{plot_type}.html"])

    assert f"Saved: plot{plot_type}.html" in result["out"]
    body = (tmp_path / f"plot{plot_type}.html").read_text(encoding="utf-8")
    assert "plotly" in body.lower()
    assert not result["leaked"]


def test_an_unknown_plot_type_is_reported(flow):
    result = flow("d", ["9"])

    assert "Invalid plot type." in result["out"]


def test_the_plot_generator_leaves_the_global_rng_alone(flow):
    import numpy as np

    before = np.random.get_state()
    result = flow("d", ["1", "rng.html"])
    after = np.random.get_state()

    assert "Saved: rng.html" in result["out"]
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


# ── Plugin manager (f) ───────────────────────────────────────────────────────

def test_plugin_discovery_and_listing_run(flow):
    assert "Discovered 0 plugin(s)" in flow("f", ["1"])["out"]
    listing = flow("f", ["2"])["out"]
    assert "Plugin Manager" in listing
    assert "Traceback" not in listing


def test_plugin_template_is_created_in_the_working_directory(flow, tmp_path):
    result = flow("f", ["3", "demo-tool"])

    assert "Created:" in result["out"]
    assert (tmp_path / "biosuite-plugin-demo-tool").is_dir()


# ── Domain flows (a, b, c, 95, 97) ───────────────────────────────────────────

def test_16s_classification_reports_a_composition(flow):
    result = flow("a", ["ecoli:TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGG", ""])

    assert "Taxonomic Composition:" in result["out"]


def test_sv_cnv_detection_runs_on_the_demo_coverage(flow):
    result = flow("b", ["y"])

    assert "bins" in result["out"]
    assert "Traceback" not in result["out"]


def test_sv_cnv_without_a_file_falls_back_loudly(flow):
    """The fallback is allowed; being quiet about it is not."""
    result = flow("b", ["n", "absent.csv"])

    assert "File not found. Using demo data." in result["out"]
    assert "bins" in result["out"]


def test_sv_cnv_demo_leaves_the_global_rng_alone(flow):
    import numpy as np

    before = np.random.get_state()
    result = flow("b", ["y"])
    after = np.random.get_state()

    assert "bins" in result["out"]
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_bigwig_reader_names_a_missing_file(flow):
    assert "File not found." in flow("c", ["absent.bw"])["out"]


def test_go_browser_answers_a_query(flow):
    result = flow("95", ["cell"])

    assert "GO:" in result["out"]
    assert "Found" in result["out"]


def test_gwas_from_a_file_uses_the_file(flow, tmp_path):
    (tmp_path / "snps.csv").write_text(
        "chrom,pos,snp_id,case_alt,case_ref,ctrl_alt,ctrl_ref\n"
        "chr1,1000,rsLead,90,10,10,90\nchr1,2000,rsNull,50,50,50,50\n", encoding="utf-8")

    result = flow("97", ["n", "snps.csv"])

    assert "rsLead" in result["out"]
    assert "GWAS" in result["out"].upper()


def test_gwas_refuses_a_missing_file_instead_of_inventing_snps(flow):
    result = flow("97", ["n", "absent.csv"])

    assert "File not found: absent.csv" in result["out"]
    assert "Genome-wide significant" not in result["out"], \
        "the option substituted 2,000 simulated SNPs and said nothing about it"


def test_gwas_without_a_path_says_so(flow):
    result = flow("97", ["n", ""])

    assert "No file given" in result["out"]


# ── row-entry flows (58, 62, 123, 124, 125, 77, 78, 79, 80) ──────────────────

def test_popgen_full_analysis_on_entered_genotype_rows(flow):
    result = flow("58", ["0,1,2", "1,1,0", "2,2,1", "0,0,1", ""])

    assert "Population Genetics Report" in result["out"]
    assert not result["leaked"]


@pytest.mark.parametrize("model", ["RF", "SVM"])
def test_ml_training_on_entered_rows(flow, model):
    result = flow("62", ["1.0,2.0", "2.0,1.0", "3.0,4.0", "4.0,3.0", "",
                         "a,a,b,b", model])

    assert "Machine Learning Report" in result["out"]
    assert "Model:" in result["out"]
    assert "Traceback" not in result["out"]


def test_survival_curve_from_entered_times_and_events(flow):
    result = flow("123", ["1,2,3,4,5", "1,0,1,0,1"])

    assert result["out"].strip()
    assert "Traceback" not in result["out"]


def test_ppi_network_counts_nodes_and_edges(flow):
    result = flow("124", ["A,B,1.0", "B,C,2.0", "C,A,1.5", ""])

    assert "Nodes:" in result["out"]
    assert not result["leaked"], "option 124 used to leave its figure registered"


def test_network_statistics_without_plotting(flow):
    result = flow("125", ["A,B,1.0", "B,C,2.0", ""])

    assert "nodes" in result["out"] and "edges" in result["out"]


def test_upset_from_entered_sets(flow):
    result = flow("77", ["SetA,gene1,gene2,gene3", "SetB,gene2,gene3,gene4", ""])

    assert not result["leaked"]
    assert "No sets entered" not in result["out"]


def test_upset_without_sets_says_so(flow):
    assert "No sets entered." in flow("77", [""])["out"]


def test_genome_browser_tracks_from_real_files(flow):
    result = flow("78", ["bed", "regions.bed", "variant", "absent.vcf", ""])

    assert "No tracks added" not in result["out"]
    assert not result["leaked"]


def test_genome_browser_without_tracks_says_so(flow):
    assert "No tracks added." in flow("78", [""])["out"]


def test_interactive_scatter_from_named_columns(flow):
    result = flow("79", ["y", "points.csv", "x", "y"])

    assert "Saved: scatter_interactive.html" in result["out"]
    assert "scatter_interactive.html" in result["files"]


def test_interactive_scatter_rejects_columns_that_are_not_there(flow):
    assert "Invalid columns." in flow("79", ["y", "points.csv", "q1", "q2"])["out"]


def test_interactive_scatter_reports_a_missing_file(flow):
    assert "File not found." in flow("79", ["y", "absent.csv"])["out"]


def test_interactive_scatter_falls_back_to_demo_data(flow):
    result = flow("79", ["n"])

    assert "Saved: scatter_interactive.html" in result["out"]


def test_interactive_scatter_demo_leaves_the_global_rng_alone(flow):
    import numpy as np

    before = np.random.get_state()
    result = flow("79", ["n"])
    after = np.random.get_state()

    assert "Saved: scatter_interactive.html" in result["out"]
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_an_unknown_menu_option_is_reported(flow):
    result = flow("zzz", [])

    assert "Invalid option" in result["out"]


def test_enrichment_refuses_a_method_it_does_not_have(flow):
    result = flow("35", ["BRCA1,TP53", "banana"])

    assert "Unknown method 'BANANA'" in result["out"]
    assert "use ORA, GSEA or KEGG" in result["out"]


def test_logo_from_entered_sequences(flow):
    result = flow("80", ["ACGTACGT", "ACGAACGT", ""])

    assert not result["leaked"]
    assert "No sequences" not in result["out"]


# ── file-driven flows (20, 50) ───────────────────────────────────────────────

def test_blast_option_reports_hits_from_two_fastas(flow, tmp_path):
    (tmp_path / "query.fasta").write_text(">query\nACGTACGTACGTACGT\n", encoding="utf-8")
    (tmp_path / "db.fasta").write_text(">subject\nACGTACGTACGTACGT\n", encoding="utf-8")

    result = flow("20", ["query.fasta", "db.fasta"])

    assert "Total hits:" in result["out"]
    assert "subject" in result["out"]


def test_single_cell_reports_the_load_error(flow):
    result = flow("50", ["absent.h5ad"])

    assert "Error:" in result["out"]


# ── the remaining branch tails ───────────────────────────────────────────────

def test_upgma_tree_names_a_file_without_sequences(flow, tmp_path):
    (tmp_path / "empty.fasta").write_text("", encoding="utf-8")

    assert "No sequences read from that file." in flow("44", ["empty.fasta"])["out"]


def test_gwas_demo_path_reports_its_own_numbers(flow):
    result = flow("97", ["y"])

    assert "Genome-wide significant" in result["out"]


def test_genome_browser_reads_a_variant_track(flow, tmp_path):
    (tmp_path / "variants.vcf").write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t150\trs1\tA\tG\t99\tPASS\t.\n", encoding="utf-8")

    result = flow("78", ["variant", "variants.vcf", ""])

    assert "No tracks added" not in result["out"]
    assert not result["leaked"]


def test_sv_cnv_from_a_coverage_file(flow, tmp_path):
    (tmp_path / "cov.csv").write_text("\n".join("30" for _ in range(200)), encoding="utf-8")

    result = flow("b", ["n", "cov.csv"])

    assert "bins" in result["out"]


def test_digest_prints_the_report(flow):
    result = flow("m", [DNA, "EcoRI"])

    assert "DIGESTION REPORT" in result["out"]


def test_primer_option_designs_and_amplifies(flow):
    result = flow("o", [DNA, ""])

    assert "PRIMER DESIGN REPORT" in result["out"]
    assert "PCR AMPLIFICATION REPORT" in result["out"]


def test_ligation_option_reports_the_product(flow):
    """``simulate_ligation`` takes sequences, not lengths — the prompt said otherwise."""
    result = flow("q", ["ACGTACGT", "GGCCGGCC", ""])

    assert "LIGATION REPORT" in result["out"]
    assert "Product size: 16 bp" in result["out"]


def test_ligation_option_without_fragments_says_so(flow):
    assert "No fragments provided." in flow("q", [""])["out"]


def test_provenance_keeps_json_that_is_not_an_object(flow, tmp_path):
    """``[1,2]`` is valid JSON but not a parameter mapping — store it, don't drop it."""
    flow("e", ["2", "flow.db", "sequence", "gc_content", "[1, 2]", "note"])
    flow("e", ["5", "flow.db", "params.json"])

    exported = json.loads((tmp_path / "params.json").read_text(encoding="utf-8"))
    assert exported["total_steps"] == 1
    assert exported["steps"][0]["params"] == {"value": [1, 2]}


def test_provenance_reports_an_unknown_action(flow):
    assert "Not an action: '7'" in flow("e", ["7"])["out"]


def test_the_menu_returns_after_a_cancelled_action(monkeypatch):
    """Ctrl-C during an action goes back to the menu; it must not kill the session."""
    import sys

    calls = []

    def interrupt(prompt=""):
        calls.append(prompt)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return "1" if len(calls) == 1 else "0"

    monkeypatch.setattr(builtins, "input", interrupt)
    monkeypatch.setattr(sys, "argv", ["biosuite"])
    monkeypatch.setitem(menu.config, "quiet", True)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert menu.main_cli() == 0
    assert "Cancelled" in ANSI.sub("", buffer.getvalue())


def test_the_menu_propagates_an_explicit_exit(monkeypatch):
    import sys

    calls = []

    def answer(prompt=""):
        calls.append(prompt)
        if len(calls) == 1:
            return "1"                      # volcano_plot(), patched below
        raise SystemExit(3)

    monkeypatch.setattr(builtins, "input", answer)
    monkeypatch.setattr(sys, "argv", ["biosuite"])
    # The demos are imported inside their menu branches, so the interception
    # point is the source module, not the menu.
    monkeypatch.setattr(biological_plots, "volcano_plot",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit(3)))

    with pytest.raises(SystemExit) as exit_info:
        menu.main_cli()

    assert exit_info.value.code == 3


def test_debug_mode_shows_a_traceback_and_keeps_the_session(monkeypatch):
    """``BIOSUITE_DEBUG=1`` adds the traceback; it does not end the session (BSU-014)."""
    import sys

    calls = []

    def answer(prompt=""):
        calls.append(prompt)
        if len(calls) == 1:
            return "2"                      # the PCA plot action
        return "0"

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(builtins, "input", answer)
    monkeypatch.setattr(sys, "argv", ["biosuite"])
    monkeypatch.setattr(biological_plots, "pca_plot", boom)
    monkeypatch.setenv("BIOSUITE_DEBUG", "1")

    buffer, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(errors):
        assert menu.main_cli() == 0
    rendered = ANSI.sub("", buffer.getvalue() + errors.getvalue())

    assert "Traceback (most recent call last)" in rendered, \
        "traceback.print_exc() writes to stderr; both streams belong to the user"
    assert "kaboom" in rendered
    assert "Goodbye" in buffer.getvalue(), "a debugging aid must not take the session down"


def test_single_cell_option_prints_the_pipeline_report(monkeypatch, flow):
    """The option's own wiring: a load that succeeds must reach the report."""
    import biosuite.core.single_cell as single_cell

    monkeypatch.setattr(single_cell, "load_count_matrix", lambda fp: ("adata", None))
    monkeypatch.setattr(single_cell, "run_full_pipeline", lambda a: (None, {"cells": 3}))
    monkeypatch.setattr(single_cell, "format_sc_report", lambda stats: f"SC REPORT {stats}")

    result = flow("50", ["matrix.csv"])

    assert "SC REPORT" in result["out"]
    assert "Error:" not in result["out"]


def test_genome_browser_builds_a_coverage_track_from_a_sam(flow, tmp_path):
    (tmp_path / "reads.sam").write_text(
        "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n"
        "r1\t0\tchr1\t100\t60\t10M\t*\t0\t0\tACGTACGTAC\tIIIIIIIIII\n", encoding="utf-8")

    result = flow("78", ["coverage", "reads.sam", ""])

    assert "No tracks added" not in result["out"]
    assert not result["leaked"]


def test_bigwig_option_lists_its_chromosomes(monkeypatch, flow, tmp_path):
    """The reader's own branches are covered elsewhere; this pins the CLI's rendering."""
    import biosuite.core.file_formats as file_formats

    (tmp_path / "x.bw").write_bytes(b"\x00" * 64)
    monkeypatch.setattr(file_formats, "read_bigwig",
                        lambda fp, **kw: {"chroms": {"chr1": [1, 2, 3], "chr2": [4]}})

    result = flow("c", ["x.bw"])

    assert "Chromosomes: 2" in result["out"]
    assert "chr1: 3 values" in result["out"]


def test_bigwig_option_prints_a_summary_for_another_shape(monkeypatch, flow, tmp_path):
    import biosuite.core.file_formats as file_formats

    (tmp_path / "y.bw").write_bytes(b"\x00" * 64)
    monkeypatch.setattr(file_formats, "read_bigwig", lambda fp, **kw: {"values": [1, 2]})
    monkeypatch.setattr(file_formats, "bigwig_summary", lambda fp: {"mean": 1.5})
    monkeypatch.setattr(file_formats, "format_bigwig_summary",
                        lambda summary: f"SUMMARY mean={summary['mean']}")

    assert "SUMMARY mean=1.5" in flow("c", ["y.bw"])["out"]


def test_bigwig_option_shows_the_reader_error(monkeypatch, flow, tmp_path):
    import biosuite.core.file_formats as file_formats

    (tmp_path / "broken.bw").write_bytes(b"\x00" * 64)
    monkeypatch.setattr(file_formats, "read_bigwig",
                        lambda fp, **kw: {"error": "truncated header"})

    assert "Error: truncated header" in flow("c", ["broken.bw"])["out"]


def test_the_api_action_survives_ctrl_c(monkeypatch):
    def interrupt(port=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(menu, "_launch_api", interrupt)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "0")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        menu._handle_menu_choice("90")

    assert "API server stopped." in ANSI.sub("", buffer.getvalue())


def test_the_gui_entry_point_runs_the_app(monkeypatch):
    import biosuite.gui.main_window as main_window

    started = []
    monkeypatch.setattr(main_window, "BioSuiteApp",
                        lambda: SimpleNamespace(mainloop=lambda: started.append("run")))

    menu.main_cli_gui()

    assert started == ["run"]


# ── tails that only one specific answer sequence reaches ──────────────────────
# The two sweeps answer every prompt with the same generic text, so they always
# take the "user gave me something usable" branch.  These are the paths behind
# it: a file that parses to nothing, a line in the wrong shape, a track type
# that does not exist, an action outside the menu's own list.

def test_msa_option_names_a_file_without_records(flow, tmp_path):
    (tmp_path / "empty.fasta").write_text("", encoding="utf-8")

    assert "No sequences read from that file." in flow("21", ["empty.fasta"])["out"]


def test_upset_ignores_a_line_that_is_not_a_set(flow):
    result = flow("77", ["SetA,gene1,gene2", "just_a_name", ""])

    assert "Ignored line 'just_a_name'" in result["out"]
    assert "No sets entered" not in result["out"]
    assert not result["leaked"]


@pytest.mark.parametrize("track,path,message", [
    ("bed", "absent.bed", "BED file not found; track skipped."),
    ("variant", "absent.vcf", "VCF file not found; track skipped."),
    ("coverage", "absent.bam", "BAM/SAM file not found; track skipped."),
])
def test_genome_browser_names_a_track_whose_file_is_not_there(flow, track, path, message):
    result = flow("78", [track, path, ""])

    assert message in result["out"]
    assert "No tracks added." in result["out"]


def test_genome_browser_rejects_an_unknown_track_type(flow):
    assert "Unknown track type 'wig'" in flow("78", ["wig", ""])["out"]


def test_synteny_dotplot_uses_the_gene_orders_that_were_entered(flow):
    """Empty answers make the option fall back to its demo genomes; real answers
    must not, and the two data sets score differently, which is what makes this
    test able to tell the branches apart."""
    from biosuite.plotting.synteny import compute_synteny_score

    demo = (["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"],
            ["GeneA", "GeneC", "GeneB", "GeneE", "GeneD"])
    entered = (["ALB", "ADH1", "ACT1", "CDC28"], ["ALB", "ACT1", "ADH1", "CDC28"])

    result = flow("81", [",".join(entered[0]), ",".join(entered[1])])

    assert f"Synteny score: {compute_synteny_score(*entered)[0]:.3f}" in result["out"]
    assert f"Synteny score: {compute_synteny_score(*demo)[0]:.3f}" not in result["out"]
    assert not result["leaked"]


def test_16s_input_ignores_a_sequence_without_a_name(flow):
    result = flow("a", ["unnamed_sequence", f"Ecoli:{DNA}", ""])

    assert "Ignored 'unnamed_sequence'" in result["out"]
    assert "Using demo data" not in result["out"]


def test_plugin_manager_reports_an_unknown_action(flow):
    assert "Unknown action '9'." in flow("f", ["9"])["out"]


def test_plugin_template_needs_a_name(flow):
    result = flow("f", ["3", ""])

    assert "No plugin name given." in result["out"]
    assert "Created:" not in result["out"]


def test_file_inspector_reports_a_file_it_cannot_parse(flow, tmp_path):
    (tmp_path / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")

    assert "no records to list" in flow("k", ["notes.txt"])["out"]


# ── Tier 2 feature flows: enrichment 35, variants 41, peaks 42, FBA 57, epigenomics 59 ──


def test_enrichment_kegg_method_reaches_the_core(flow, monkeypatch):
    from biosuite.core import enrichment

    captured = {}
    real = enrichment.run_kegg_enrichment

    def spy(genes, **kwargs):
        captured["genes"] = genes
        return real(genes, **kwargs)

    monkeypatch.setattr(enrichment, "run_kegg_enrichment", spy)

    result = flow("35", ["BRCA1, TP53, EGFR", "KEGG"])

    assert captured["genes"] == ["BRCA1", "TP53", "EGFR"]
    expected = enrichment.format_enrichment_report(real(["BRCA1", "TP53", "EGFR"]))
    assert expected.strip() in result["out"]


def test_enrichment_default_method_is_still_ora(flow):
    from biosuite.core.enrichment import format_enrichment_report, run_ora

    result = flow("35", ["BRCA1,TP53", ""])

    expected = format_enrichment_report(run_ora(["BRCA1", "TP53"]))
    assert expected.strip() in result["out"]


def test_enrichment_gsea_method_reaches_the_core(flow):
    from biosuite.core.enrichment import format_enrichment_report, run_gsea

    result = flow("35", ["BRCA1,TP53", "GSEA"])

    expected = format_enrichment_report(run_gsea(["BRCA1", "TP53"]))
    assert expected.strip() in result["out"]


def test_enrichment_unknown_method_is_named(flow):
    result = flow("35", ["BRCA1", "BLAST"])

    assert "Unknown method 'BLAST': use ORA, GSEA or KEGG." in result["out"]


def _variant_report():
    from biosuite.core.variant_calling import Variant, VariantReport

    def variant(pos, quality, depth, alt_count):
        return Variant(chrom="chr1", pos=pos, ref="A", alt="G", quality=quality,
                       depth=depth, alt_count=alt_count, genotype="0/1",
                       variant_type="SNP")

    variants = [variant(100, 50.0, 20, 10),      # passes the default filter
                variant(200, 5.0, 20, 10),       # quality below 20
                variant(300, 50.0, 2, 1)]        # depth below 5
    return VariantReport(tool="builtin", engine="builtin", total_variants=3,
                         snps=3, variants=variants)


def test_variant_filter_reaches_the_core(flow, monkeypatch):
    from biosuite.core import variant_calling

    report = _variant_report()
    monkeypatch.setattr(variant_calling, "call_variants", lambda *a, **k: report)
    expected = variant_calling.filter_variants(report.variants, min_quality=20,
                                               min_depth=5, max_allele_freq=0.95)
    assert len(expected) == 1, "the crafted report must discriminate the filter"

    result = flow("41", ["reads.sam", "20,5,0.95"])

    assert variant_calling.variant_summary(report).strip() in result["out"]
    assert f"Kept {len(expected)} of {len(report.variants)} variant(s) after filtering." \
        in result["out"]


def test_variant_filter_blank_keeps_everything(flow, monkeypatch):
    from biosuite.core import variant_calling

    monkeypatch.setattr(variant_calling, "call_variants",
                        lambda *a, **k: _variant_report())

    result = flow("41", ["reads.sam", ""])

    assert "after filtering" not in result["out"]


def test_variant_filter_bad_numbers_are_named(flow, monkeypatch):
    from biosuite.core import variant_calling

    monkeypatch.setattr(variant_calling, "call_variants",
                        lambda *a, **k: _variant_report())

    result = flow("41", ["reads.sam", "20,5"])

    assert "Filter needs three numbers: min_quality,min_depth,max_af." in result["out"]
    assert "after filtering" not in result["out"]


def _peak_report(peaks):
    from biosuite.core.peak_calling import PeakReport

    return PeakReport(engine="builtin", total_peaks=len(peaks), peaks=peaks)


def _peaks():
    from biosuite.core.peak_calling import Peak

    return [Peak(chrom="chr1", start=150, end=250, summit=200, score=50.0,
                 p_value=0.001, fold_enrichment=5.0),
            Peak(chrom="chr1", start=500, end=600, summit=550, score=40.0,
                 p_value=0.002, fold_enrichment=4.0)]


def test_peak_annotation_names_the_nearest_gene(flow, monkeypatch):
    import copy

    from biosuite.core import peak_calling

    peaks = _peaks()
    monkeypatch.setattr(peak_calling, "call_peaks",
                        lambda *a, **k: _peak_report(peaks))
    expected = peak_calling.annotate_peaks_with_genes(copy.deepcopy(peaks), "regions.bed")
    assert [p.name for p in expected] == ["geneA", "geneB"], \
        "regions.bed must place one peak inside geneA and one nearest geneB"

    result = flow("42", ["chip.sam", "regions.bed"])

    for p in expected:
        assert f"{p.chrom}:{p.start}-{p.end} -> {p.name}" in result["out"]


def test_peak_annotation_blank_skips(flow, monkeypatch):
    from biosuite.core import peak_calling

    monkeypatch.setattr(peak_calling, "call_peaks",
                        lambda *a, **k: _peak_report(_peaks()))

    result = flow("42", ["chip.sam", ""])

    assert "->" not in result["out"]


def test_peak_annotation_without_peaks_says_so(flow, monkeypatch):
    from biosuite.core import peak_calling

    monkeypatch.setattr(peak_calling, "call_peaks",
                        lambda *a, **k: _peak_report([]))

    result = flow("42", ["chip.sam", "regions.bed"])

    assert "No peaks to annotate." in result["out"]


def test_peak_annotation_names_an_unreadable_bed(flow, monkeypatch):
    from biosuite.core import peak_calling

    monkeypatch.setattr(peak_calling, "call_peaks",
                        lambda *a, **k: _peak_report(_peaks()))

    result = flow("42", ["chip.sam", "absent.bed"])

    assert "Could not read the gene BED" in result["out"]


def test_knockout_without_cobra_says_no_gene_was_tested(flow, monkeypatch):
    from biosuite.core import metabolism

    monkeypatch.setattr(metabolism, "HAS_COBRA", False)

    result = flow("57", ["", "b0001,b0002"])

    assert "No gene was tested: gene knockouts need COBRApy" in result["out"]
    assert "essential=" not in result["out"]


def test_knockout_results_come_from_the_core(flow, monkeypatch, tmp_path):
    from biosuite.core import metabolism

    (tmp_path / "model.xml").write_text(
        '<?xml version="1.0"?><sbml level="2" version="1"><model id="m"/></sbml>\n',
        encoding="utf-8")
    results = [metabolism.KnockoutResult(gene="b0001", wild_type_flux=0.5,
                                         knockout_flux=0.0, growth_reduction=1.0,
                                         essential=True),
               metabolism.KnockoutResult(gene="b0002", wild_type_flux=0.5,
                                         knockout_flux=None, growth_reduction=None,
                                         essential=False)]
    monkeypatch.setattr(metabolism, "HAS_COBRA", True)
    monkeypatch.setattr(metabolism, "_cobra_knockout", lambda *a, **k: (results, []))

    result = flow("57", ["model.xml", "b0001,b0002"])

    assert "b0001: knockout flux=0.0000  essential=True" in result["out"]
    assert "b0002: knockout flux=n/a  essential=False" in result["out"]


def test_knockout_empty_results_with_cobra_present_are_reported(flow, monkeypatch, tmp_path):
    from biosuite.core import metabolism

    (tmp_path / "model.xml").write_text(
        '<?xml version="1.0"?><sbml level="2" version="1"><model id="m"/></sbml>\n',
        encoding="utf-8")
    monkeypatch.setattr(metabolism, "HAS_COBRA", True)
    monkeypatch.setattr(metabolism, "_cobra_knockout", lambda *a, **k: ([], []))

    result = flow("57", ["model.xml", "b0001"])

    assert "Knockout analysis returned no results." in result["out"]


def test_knockout_blank_skips(flow):
    result = flow("57", ["", ""])

    assert "knockout" not in result["out"].lower()
    assert "Flux" in result["out"] or "objective" in result["out"].lower()


def _bisulfite_bed(path, rows):
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in rows),
                    encoding="utf-8")


def test_epigenomics_dmr_mode_finds_the_core_dmrs(flow, tmp_path):
    from biosuite.core.epigenomics import find_dmrs, parse_bisulfite_bed

    _bisulfite_bed(tmp_path / "g1.bed",
                   [("chr1", 100, 101, 18, 20, "CpG"), ("chr1", 200, 201, 10, 20, "CpG")])
    _bisulfite_bed(tmp_path / "g2.bed",
                   [("chr1", 100, 101, 2, 20, "CpG"), ("chr1", 200, 201, 10, 20, "CpG")])
    expected = find_dmrs(parse_bisulfite_bed(str(tmp_path / "g1.bed")),
                         parse_bisulfite_bed(str(tmp_path / "g2.bed")))
    assert len(expected) == 1, "only the 0.9-vs-0.1 site may pass as a DMR"

    result = flow("59", ["DMR", "g1.bed", "g2.bed"])

    assert f"DMRs found: {len(expected)}" in result["out"]
    d = expected[0]
    assert f"{d['chrom']}:{d['pos']}" in result["out"]
    assert f"delta={d['delta_methylation']:+.4f}" in result["out"]


def test_epigenomics_dmr_mode_names_an_unreadable_file(flow):
    result = flow("59", ["DMR", "absent.bed", ""])

    assert "Could not read a BED file" in result["out"]


def test_epigenomics_atac_mode_reports_the_core_stats(flow, tmp_path):
    from biosuite.core.epigenomics import atac_peak_stats, parse_atac_peaks

    (tmp_path / "peaks.bed").write_text(
        "track name=atac\n"
        "chr1\t100\t200\tpeak1\t50\n"
        "chr1\t300\t450\tpeak2\t80\n"
        "chr2\t100\t250\tpeak3\t60\n"
        "bad\tline\there\n", encoding="utf-8")
    stats = atac_peak_stats(parse_atac_peaks(str(tmp_path / "peaks.bed")))
    assert stats["total_peaks"] == 3, "the track and malformed lines must be skipped"

    result = flow("59", ["ATAC", "peaks.bed"])

    assert f"Peaks: {stats['total_peaks']}  Total bp: {stats['total_bp']}" in result["out"]
    assert f"Mean length: {stats['mean_length']:.1f}" in result["out"]
    assert f"Chromosomes: {stats['chromosomes']}" in result["out"]


def test_epigenomics_atac_mode_names_a_file_without_peaks(flow, tmp_path):
    (tmp_path / "empty.bed").write_text("# nothing here\n", encoding="utf-8")

    result = flow("59", ["ATAC", "empty.bed"])

    assert "No peaks parsed from that file." in result["out"]


def test_epigenomics_atac_mode_names_an_unreadable_file(flow):
    result = flow("59", ["ATAC", "absent.bed"])

    assert "Could not read the peak file" in result["out"]


def test_epigenomics_dmr_mode_truncates_a_long_list(flow, tmp_path):
    rows1 = [("chr1", 100 + i * 10, 101 + i * 10, 18, 20, "CpG") for i in range(21)]
    rows2 = [("chr1", 100 + i * 10, 101 + i * 10, 2, 20, "CpG") for i in range(21)]
    _bisulfite_bed(tmp_path / "g1.bed", rows1)
    _bisulfite_bed(tmp_path / "g2.bed", rows2)

    result = flow("59", ["DMR", "g1.bed", "g2.bed"])

    assert "DMRs found: 21" in result["out"]
    assert "... and 1 more." in result["out"]


def test_peak_annotation_truncates_a_long_list(flow, monkeypatch):
    from biosuite.core import peak_calling
    from biosuite.core.peak_calling import Peak

    peaks = [Peak(chrom="chr1", start=1000 + i * 100, end=1050 + i * 100,
                  summit=1025 + i * 100, score=30.0, p_value=0.01,
                  fold_enrichment=3.0) for i in range(21)]
    monkeypatch.setattr(peak_calling, "call_peaks",
                        lambda *a, **k: _peak_report(peaks))

    result = flow("42", ["chip.sam", "regions.bed"])

    assert "... and 1 more." in result["out"]


def test_epigenomics_default_mode_is_still_methylation(flow, tmp_path):
    from biosuite.core.epigenomics import (
        calculate_methylation_levels,
        format_epigenomics_report,
        parse_bisulfite_bed,
    )

    _bisulfite_bed(tmp_path / "meth.bed",
                   [("chr1", 100, 101, 18, 20, "CpG"), ("chr1", 200, 201, 10, 20, "CHG")])
    expected = format_epigenomics_report(
        calculate_methylation_levels(parse_bisulfite_bed(str(tmp_path / "meth.bed"))))

    result = flow("59", ["", "meth.bed"])

    assert expected.strip() in result["out"]


def test_epigenomics_unknown_mode_is_named(flow):
    result = flow("59", ["BLAST"])

    assert "Unknown mode 'BLAST': use METH, DMR or ATAC." in result["out"]

