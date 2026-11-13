"""The Transcriptomics tab: expression, trimming and quantification handlers.

The handlers read an entry, call the ``biosuite.core`` module behind the tab and put
that module's own output into the results box, so the expected report, table and
status line are all recomputed here from ``expression``, ``trimming`` and
``quantification`` — the tab is graded against the code it wraps.  The Browse button
is driven as an actual click so the file-picker wiring and its cancel path are both
covered, and every guard is tested with a value the handler cannot ignore.
"""
import customtkinter as ctk
import numpy as np
import os
import pandas as pd
import pytest

from biosuite.core.expression import (
    calculate_effect_size,
    calculate_fold_change,
    differential_expression,
    read_counts_matrix,
)
from biosuite.core.quantification import format_quant_report, quantify_reads
from biosuite.core.trimming import format_trim_report, trim_fastq

GENES = [f"GENE{i}" for i in range(8)]


def _button_in(frame, text):
    """The tab's own button, found the way a user finds it: by its label."""
    stack = [frame]
    while stack:
        widget = stack.pop()
        if isinstance(widget, ctk.CTkButton) and widget.cget("text") == text:
            return widget
        stack.extend(widget.winfo_children())
    raise AssertionError(f"no {text!r} button in {frame!r}")


@pytest.fixture
def counts_csv(tmp_path):
    """Four samples over two conditions, four genes strongly up-regulated."""
    rng = np.random.default_rng(11)
    control = rng.integers(80, 140, size=(len(GENES), 2)).astype(float)
    treat = rng.integers(300, 600, size=(len(GENES), 2)).astype(float)
    treat[:4] *= 6                       # a block of genes that must come out significant
    treat[4:] *= 0.2                     # and a block that must not
    data = np.hstack([control, treat])
    path = tmp_path / "counts.csv"
    frame = pd.DataFrame(data, columns=["Sample_A", "Sample_B", "Sample_C", "Sample_D"])
    frame.insert(0, "gene", GENES)
    frame.to_csv(path, index=False)
    return path


def _trimmed(path):
    """The output path the tab derives from the input path."""
    return os.path.splitext(str(path))[0] + "_trimmed.fastq"


@pytest.fixture
def fastq_file(tmp_path):
    """Four reads, the last bases of each given Q0 quality so they get trimmed."""
    adapter = "AGATCGGAAGAGC"
    lines = []
    for i in range(4):
        seq = f"ACGT{adapter}TTGACCGATCAAGGCTTAAGCCGGTTAACG{i}"[:60]
        qual = "I" * (len(seq) - 6) + "!" * 6
        lines.extend([f"@read{i}", seq, "+", qual])
    path = tmp_path / "sample.fastq"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def transcriptome_fasta(tmp_path):
    """Three transcripts long enough for the built-in 31-mer engine."""
    seqs = []
    for i in range(3):
        seq = (f"ACGTGGAATTCCGGTAAGCTCGATCGATCGATCGATCGGATTCA{i}"
               f"AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT")
        seqs.append((f"TRANSCRIPT_{i}", seq))
    path = tmp_path / "transcriptome.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in seqs), encoding="utf-8")
    return path, seqs


@pytest.fixture
def quant_inputs(tmp_path, transcriptome_fasta):
    """A FASTQ whose reads are taken from the transcripts, so they must map."""
    fasta, seqs = transcriptome_fasta
    lines = []
    for i, (_name, seq) in enumerate(seqs):
        for copy in range(2):
            read = seq[copy * 10: copy * 10 + 75]
            lines.extend([f"@r{i}_{copy}", read, "+", "I" * len(read)])
    reads = tmp_path / "reads.fastq"
    reads.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reads, fasta


@pytest.fixture(autouse=True)
def _default_entries(ui):
    """The app is session-scoped, so a value typed by one test would leak into the next."""
    ui.set("expr_conds", "control,control,treat,treat")
    ui.set("trim_qual", "20")
    ui.set("trim_minlen", "36")


# ─── Browse button ───────────────────────────────────────────────────────────

def test_browse_button_puts_the_chosen_file_in_its_own_entry(ui, counts_csv):
    ui.open_path = str(counts_csv)
    ui.clear("expr_path")

    _button_in(ui.app.frames["expression"], "Browse").invoke()

    assert ui.text("expr_path") == str(counts_csv)
    assert ui.text("trim_path") != str(counts_csv), "the expression button fills its own entry only"


def test_a_cancelled_browse_leaves_the_entry_alone(ui):
    ui.open_path = ""
    ui.set("trim_path", "keep/me.fastq")

    _button_in(ui.app.frames["trimming"], "Browse").invoke()

    assert ui.text("trim_path") == "keep/me.fastq"
    assert ui.messages == []


# ─── Differential expression ─────────────────────────────────────────────────

def test_run_expr_without_a_matrix_asks_for_one(ui):
    ui.clear("expr_path")
    ui.clear("expr_result")

    ui.app._run_expr()

    assert ui.messages_of("warning") == ["Select a count matrix first."]
    assert ui.text("expr_result").strip() == ""


def test_run_expr_without_conditions_asks_for_them(ui, counts_csv):
    ui.set("expr_path", str(counts_csv))
    ui.clear("expr_conds")
    ui.clear("expr_result")

    ui.app._run_expr()

    assert ui.messages_of("warning") == ["Enter conditions."]
    assert ui.text("expr_result").strip() == ""


def test_run_expr_reports_a_matrix_it_could_not_read(ui, tmp_path):
    ui.set("expr_path", str(tmp_path / "absent.csv"))
    ui.set("expr_conds", "control,control,treat,treat")
    ui.clear("expr_result")

    ui.app._run_expr()

    assert ui.messages_of("error") == ["Failed to read counts matrix."]
    assert ui.text("expr_result").strip() == ""


def test_run_expr_refuses_conditions_that_do_not_match_the_columns(ui, counts_csv):
    """Three labels cannot describe a four-sample matrix."""
    expected = "Conditions (3) != numeric columns (4)."
    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "control,treat,treat")
    ui.clear("expr_result")

    ui.app._run_expr()

    assert ui.messages_of("error") == [expected]
    assert ui.text("expr_result").strip() == ""


def test_run_expr_refuses_a_blank_condition_label(ui, counts_csv):
    """The core accepts two groups even when one of them is unnamed, so the blank
    label would silently become a real condition; the tab must refuse it instead."""
    counts = read_counts_matrix(str(counts_csv))
    de = differential_expression(counts, ["control", "control", "", ""])
    assert len(de) == len(counts), "positive control: the core does not refuse an unnamed group"

    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "control,control,,")
    ui.set("expr_result", "sentinel table")

    ui.app._run_expr()

    assert ui.messages_of("error") == ["Condition labels must not be blank."]
    assert ui.text("expr_result") == "sentinel table\n"


def test_run_expr_shows_the_table_the_core_computed(ui, counts_csv):
    counts = read_counts_matrix(str(counts_csv))
    conditions = ["control", "control", "treat", "treat"]
    de = differential_expression(counts, conditions)
    sig = int((de["padj"] < 0.05).sum())
    assert sig > 0, "the fixture matrix must contain differentially expressed genes"

    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", ",".join(conditions))
    ui.app._run_expr()

    assert ui.text("expr_result").strip() == \
        ("DE Results (top 20, BH-corrected):\n\n" + de.head(20).to_string()).strip()
    assert ui.status() == f"DE complete: {sig} significant genes (FDR < 0.05)"
    assert ui.messages == []


def test_run_expr_draws_a_volcano_plot_and_shows_it(ui, counts_csv):
    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "control,control,treat,treat")

    ui.app._run_expr()

    assert ui.plots, "the tab never asked for its figure to be displayed"
    title, fig = ui.plots[-1]
    assert title == "Volcano Plot - Differential Expression"
    ax = fig.axes[0]
    assert ax.get_xlabel() == "log2 Fold Change"
    assert ax.get_ylabel() == "-log10(p-value)"
    assert ax.get_title().startswith("DE Analysis (BH-corrected)")
    assert len(ax.collections) == 2, "the volcano must plot significant and other genes apart"


def test_run_expr_names_a_statistical_failure(ui, counts_csv):
    """Four columns labelled with four different conditions: the core refuses, and
    the tab must pass that refusal on instead of showing a stale table."""
    counts = read_counts_matrix(str(counts_csv))
    try:
        differential_expression(counts, ["a", "b", "c", "d"])
    except ValueError as exc:
        expected = str(exc)
    else:
        raise AssertionError("differential_expression accepted four groups")

    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "a,b,c,d")
    ui.set("expr_result", "sentinel table")

    ui.app._run_expr()

    assert ui.messages_of("error") == [expected]
    assert ui.text("expr_result") == "sentinel table\n"


# ─── Fold change and effect size ─────────────────────────────────────────────

CONDITIONS = ["control", "control", "treat", "treat"]


def test_fold_change_without_a_matrix_asks_for_one(ui):
    ui.clear("expr_path")
    ui.clear("expr_result")

    ui.app._run_fold_change()

    assert ui.messages_of("warning") == ["Select a count matrix first."]
    assert ui.text("expr_result").strip() == ""


def test_fold_change_without_conditions_asks_for_them(ui, counts_csv):
    ui.set("expr_path", str(counts_csv))
    ui.clear("expr_conds")
    ui.clear("expr_result")

    ui.app._run_fold_change()

    assert ui.messages_of("warning") == ["Enter conditions."]
    assert ui.text("expr_result").strip() == ""


def test_the_fold_change_table_matches_the_core(ui, counts_csv):
    counts = read_counts_matrix(str(counts_csv))
    table = calculate_fold_change(counts, CONDITIONS)
    # Positive control: the fixture's up-regulated block must be flagged.
    assert int(table["significant"].sum()) > 0
    ui.set("expr_path", str(counts_csv))
    ui.clear("expr_result")

    ui.app._run_fold_change()

    assert ui.text("expr_result").strip() == \
        ("Fold change (top 20):\n\n" + table.head(20).to_string()).strip()
    assert ui.status() == f"Fold change: {len(table)} genes"
    assert ui.messages == []


def test_fold_change_surfaces_the_cores_two_group_refusal(ui, counts_csv):
    counts = read_counts_matrix(str(counts_csv))
    with pytest.raises(ValueError) as exc:
        calculate_fold_change(counts, ["a", "b", "c", "d"])
    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "a,b,c,d")
    ui.set("expr_result", "sentinel table")

    ui.app._run_fold_change()

    assert ui.messages_of("error") == [str(exc.value)]
    assert ui.text("expr_result") == "sentinel table\n"


def test_effect_size_without_a_matrix_asks_for_one(ui):
    ui.clear("expr_path")
    ui.clear("expr_result")

    ui.app._run_effect_size()

    assert ui.messages_of("warning") == ["Select a count matrix first."]
    assert ui.text("expr_result").strip() == ""


def test_the_effect_size_table_matches_the_core(ui, counts_csv):
    counts = read_counts_matrix(str(counts_csv))
    table = calculate_effect_size(counts, CONDITIONS)
    # Positive control: every gene gets an interpretation label from the core.
    assert len(table) == len(counts)
    assert set(table["interpretation"]) <= {
        "negligible", "small", "medium", "large", "undefined"}
    ui.set("expr_path", str(counts_csv))
    ui.clear("expr_result")

    ui.app._run_effect_size()

    assert ui.text("expr_result").strip() == \
        ("Effect size (top 20):\n\n" + table.head(20).to_string()).strip()
    assert ui.status() == f"Effect size: {len(table)} genes"
    assert ui.messages == []


def test_effect_size_surfaces_the_cores_two_group_refusal(ui, counts_csv):
    counts = read_counts_matrix(str(counts_csv))
    with pytest.raises(ValueError) as exc:
        calculate_effect_size(counts, ["a", "b", "c", "d"])
    ui.set("expr_path", str(counts_csv))
    ui.set("expr_conds", "a,b,c,d")
    ui.set("expr_result", "sentinel table")

    ui.app._run_effect_size()

    assert ui.messages_of("error") == [str(exc.value)]
    assert ui.text("expr_result") == "sentinel table\n"


def test_effect_size_names_a_matrix_it_could_not_read(ui, tmp_path):
    ui.set("expr_path", str(tmp_path / "absent.csv"))
    ui.clear("expr_result")

    ui.app._run_effect_size()

    assert ui.messages_of("error") == ["Failed to read counts matrix."]
    assert ui.text("expr_result").strip() == ""


def test_fold_change_names_a_reader_that_raises(ui, counts_csv, monkeypatch):
    def broken(path):
        raise OSError("count matrix unreadable")
    monkeypatch.setattr("biosuite.core.expression.read_counts_matrix", broken)
    ui.set("expr_path", str(counts_csv))
    ui.set("expr_result", "sentinel table")

    ui.app._run_fold_change()

    assert ui.messages_of("error") == ["count matrix unreadable"]
    assert ui.text("expr_result") == "sentinel table\n"


# ─── FASTQ trimming ──────────────────────────────────────────────────────────

def test_run_trimming_without_a_file_asks_for_one(ui):
    ui.clear("trim_path")
    ui.clear("trim_result")

    ui.app._run_trimming()

    assert ui.messages_of("warning") == ["Select a FASTQ file."]
    assert ui.text("trim_result").strip() == ""


def test_run_trimming_writes_the_report_for_the_typed_thresholds(ui, fastq_file):
    out = _trimmed(fastq_file)
    report = trim_fastq(str(fastq_file), out, quality_threshold=15, min_length=20)
    assert report.reads_trimmed > 0, "the fixture reads must actually be trimmed"

    ui.set("trim_path", str(fastq_file))
    ui.set("trim_qual", "15")
    ui.set("trim_minlen", "20")
    ui.app._run_trimming()

    assert ui.text("trim_result").strip() == format_trim_report(report).strip()
    assert ui.status() == f"Trimming complete: {report.reads_trimmed} reads trimmed"
    assert os.path.exists(out), "trimming must leave its output file beside the input"
    assert ui.messages == []


def test_empty_trim_thresholds_fall_back_to_the_defaults(ui, fastq_file):
    report = trim_fastq(str(fastq_file), _trimmed(fastq_file),
                        quality_threshold=20, min_length=36)
    assert report.total_reads == 4, "all four reads are read by the same engine"

    ui.set("trim_path", str(fastq_file))
    ui.clear("trim_qual")
    ui.clear("trim_minlen")
    ui.app._run_trimming()

    assert ui.text("trim_result").strip() == format_trim_report(report).strip()
    assert ui.messages == []


def test_run_trimming_names_a_non_numeric_threshold(ui, fastq_file):
    ui.set("trim_path", str(fastq_file))
    ui.set("trim_qual", "high")
    ui.set("trim_result", "sentinel report")

    ui.app._run_trimming()

    assert ui.messages_of("error") == [
        "Quality threshold must be a whole number (got 'high')."]
    assert ui.text("trim_result") == "sentinel report\n"


def test_run_trimming_refuses_a_negative_quality(ui, fastq_file):
    ui.set("trim_path", str(fastq_file))
    ui.set("trim_qual", "-1")
    ui.set("trim_result", "sentinel report")

    ui.app._run_trimming()

    assert ui.messages_of("error") == ["Quality threshold must be zero or greater."]
    assert ui.text("trim_result") == "sentinel report\n"


def test_run_trimming_names_a_non_numeric_min_length(ui, fastq_file):
    ui.set("trim_path", str(fastq_file))
    ui.set("trim_minlen", "long")
    ui.set("trim_result", "sentinel report")

    ui.app._run_trimming()

    assert ui.messages_of("error") == [
        "Min length must be a whole number (got 'long')."]
    assert ui.text("trim_result") == "sentinel report\n"


def test_run_trimming_refuses_a_negative_min_length(ui, fastq_file):
    ui.set("trim_path", str(fastq_file))
    ui.set("trim_minlen", "-2")
    ui.set("trim_result", "sentinel report")

    ui.app._run_trimming()

    assert ui.messages_of("error") == ["Min length must be zero or greater."]
    assert ui.text("trim_result") == "sentinel report\n"


def test_a_trimming_failure_is_reported_without_touching_the_old_report(ui, fastq_file, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no quality encoding could be detected")

    monkeypatch.setattr("biosuite.core.trimming.trim_fastq", broken)
    ui.set("trim_path", str(fastq_file))
    ui.set("trim_result", "sentinel report")

    ui.app._run_trimming()

    assert ui.messages_of("error") == ["no quality encoding could be detected"]
    assert ui.text("trim_result") == "sentinel report\n"


def test_run_trimming_shows_the_core_message_for_an_absent_fastq(ui, tmp_path):
    """``trim_fastq`` reports a missing file in its report rather than raising."""
    missing = str(tmp_path / "gone.fastq")
    report = trim_fastq(missing, _trimmed(missing), quality_threshold=20, min_length=36)
    assert report.total_reads == 0 and report.message, \
        "the core reports a missing file inside its report, not as an exception"

    ui.set("trim_path", missing)
    ui.set("trim_qual", "20")
    ui.set("trim_minlen", "36")
    ui.app._run_trimming()

    assert f"File not found: {missing}" in ui.text("trim_result")
    assert ui.text("trim_result").strip() == format_trim_report(report).strip()
    assert ui.status() == "Trimming complete: 0 reads trimmed"


# ─── RNA-seq quantification ──────────────────────────────────────────────────

def test_run_quant_needs_both_inputs(ui, fastq_file):
    ui.clear("quant_reads")
    ui.clear("quant_trans")
    ui.clear("quant_result")

    ui.app._run_quant()

    assert ui.messages_of("warning") == ["Select both reads and transcriptome files."]
    assert ui.text("quant_result").strip() == ""

    ui.set("quant_reads", str(fastq_file))
    ui.messages.clear()
    ui.app._run_quant()

    assert ui.messages_of("warning") == ["Select both reads and transcriptome files."]


def test_run_quant_reports_the_transcript_count_the_core_found(ui, quant_inputs):
    reads, fasta = quant_inputs
    result = quantify_reads(str(reads), str(fasta))
    assert result.num_transcripts == 3, "the fixture transcriptome has three transcripts"

    ui.set("quant_reads", str(reads))
    ui.set("quant_trans", str(fasta))
    ui.app._run_quant()

    assert ui.text("quant_result").strip() == format_quant_report(result).strip()
    assert ui.status() == f"Quantification: {result.num_transcripts} transcripts"
    assert ui.messages == []


def test_run_quant_shows_a_core_failure(ui, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no index could be built")

    monkeypatch.setattr("biosuite.core.quantification.quantify_reads", boom)
    ui.set("quant_reads", "reads.fastq")
    ui.set("quant_trans", "transcriptome.fasta")
    ui.set("quant_result", "sentinel report")

    ui.app._run_quant()

    assert ui.messages_of("error") == ["no index could be built"]
    assert ui.text("quant_result") == "sentinel report\n"
