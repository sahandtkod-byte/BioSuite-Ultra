"""The NGS/VCF, Assembly, Metagenomics, 16S, SV-CNV and BigWig pages of one app.

Each handler reads a path entry or a text box, calls straight into
``biosuite.core`` (``ngs``, ``assembly``, ``metagenomics``, ``variant_calling``,
``file_formats``) and paints the answer.  Every expectation below is recomputed
from that same core call on the same input file the test wrote, so the boxes are
checked against the numbers the library produced rather than against a quoted
string.  Nothing here touches the network, and the one BigWig branch that needs
the absent ``bigWigReader`` library is driven through an injected stand-in, which
is called out where it is used.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core.assembly import assemble, format_assembly_report
from biosuite.core.file_formats import format_bigwig_summary
from biosuite.core.metagenomics import (
    KNOWN_TAXA,
    SILVA_16S_DB,
    analyze_diversity,
    classify_16s_rna,
    classify_reads,
    format_16s_report,
    format_metagenomics_report,
)
from biosuite.core.ngs import manhattan_from_vcf, read_vcf
from biosuite.core.variant_calling import (
    detect_cnv,
    detect_structural_variants,
    format_cnv_report,
    format_sv_report,
)

VCF_ROWS = [
    ("chr1", 1000, "rs1", "A", "G", "12.5", "PASS", "DP=10"),
    ("chr1", 2000, "rs2", "C", "T", "0.001", "PASS", "DP=8"),
    ("chr2", 500, "rs3", "G", "A", "40.0", "PASS", "DP=20"),
    ("chr2", 1500, ".", "T", "C", "3.2", "PASS", "DP=5"),
    ("chr10", 7777, "rs5", "GGTG", "G", "55.0", "PASS", "DP=12"),
]

TARGET = ("ATGCGGCCATGCAAACGTCGATTGCGATGCGCTTAGCTTAGCTTAGGGAATTCCTGGCAA"
          "CGTATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA")
KMER_ECOLI = "ATCGATCGATCGATCG"
KMER_STAPH = "GCTAGCTAGCTAGCTA"
TAXA = (KNOWN_TAXA[KMER_ECOLI], KNOWN_TAXA[KMER_STAPH])
ECOLI_16S = SILVA_16S_DB[TAXA[0]]
STAPH_16S = SILVA_16S_DB[TAXA[1]]
COVERAGE_POINTS = 2400


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "cohort.vcf"
    body = "\n".join("\t".join(str(field) for field in row) for row in VCF_ROWS)
    path.write_text("##fileformat=VCFv4.2\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + body + "\n",
                    encoding="utf-8")
    return path


@pytest.fixture
def reads_fastq(tmp_path):
    """Overlapping 31-mers tiled across ``TARGET`` in 20 bp steps."""
    reads = [(f"read{start}", TARGET[start:start + 31])
             for start in range(0, len(TARGET) - 30, 20)]
    path = tmp_path / "reads.fastq"
    path.write_text("".join(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n"
                            for name, seq in reads), encoding="utf-8")
    return path


@pytest.fixture
def taxa_fastq(tmp_path):
    """Two reads carrying a 16-mer the built-in classifier knows, one that does not."""
    records = [("@r1", "GGGG" + KMER_ECOLI + "CCCC"),
               ("@r2", "AAAC" + KMER_STAPH + "TTG"),
               ("@r3", "ACGTACGTACGTACGTACGTACGTACGTACGT")]
    path = tmp_path / "meta.fastq"
    path.write_text("".join(f"{name}\n{seq}\n+\n{'I' * len(seq)}\n"
                            for name, seq in records), encoding="utf-8")
    return path


@pytest.fixture
def coverage_csv(tmp_path):
    """Flat coverage with a dip and a bump, so both SV classes have a reason to fire."""
    values = np.full(COVERAGE_POINTS, 30.0)
    values[400:800] = 3.0
    values[1600:2000] = 90.0
    path = tmp_path / "coverage.csv"
    path.write_text("\n".join(f"{v:.1f}" for v in values) + "\n", encoding="utf-8")
    return path


# ─── NGS / VCF ──────────────────────────────────────────────────────────────

def test_the_vcf_button_needs_a_file_first(ui):
    ui.clear("vcf_path")
    ui.clear("vcf_result")

    ui.app._load_vcf()

    assert ui.messages_of("warning") == ["Select a VCF file first."]
    assert ui.text("vcf_result").strip() == ""
    assert ui.plots == []


def test_a_vcf_that_is_not_there_reports_a_failed_read(ui, tmp_path):
    missing = tmp_path / "absent.vcf"
    assert read_vcf(str(missing)) is None
    ui.set("vcf_path", str(missing))
    ui.clear("vcf_result")

    ui.app._load_vcf()

    assert ui.text("vcf_result").strip() == "Failed to read VCF."
    assert ui.messages == []
    assert ui.plots == []


def test_load_vcf_lists_the_variants_the_core_read(ui, vcf_file):
    df = read_vcf(str(vcf_file))
    assert len(df) == len(VCF_ROWS)
    ui.set("vcf_path", str(vcf_file))

    ui.app._load_vcf()

    assert ui.text("vcf_result").strip() == (
        f"Loaded {len(df)} variants.\n\n" + df.head(20).to_string())
    assert "rs5" in ui.text("vcf_result") and "GGTG" in ui.text("vcf_result")
    assert ui.status() == "VCF loaded, Manhattan plot displayed"
    assert ui.messages == []


def test_load_vcf_plots_the_manhattan_points_the_core_derived(ui, vcf_file):
    manh = manhattan_from_vcf(read_vcf(str(vcf_file)))
    ui.set("vcf_path", str(vcf_file))

    ui.app._load_vcf()

    title, fig = ui.plots[-1]
    assert title == "Manhattan Plot - VCF Analysis"
    ax = fig.axes[0]
    assert ax.get_title() == "Manhattan Plot from VCF"
    assert ax.get_xlabel() == "Position"
    assert ax.get_ylabel() == "-log10(p)"
    np.testing.assert_allclose(ax.collections[0].get_offsets(),
                               np.column_stack([manh["POS"], manh["neg_log10"]]))
    assert len(ax.collections[0].get_offsets()) == len(VCF_ROWS)
    assert ax.lines[0].get_ydata()[0] == pytest.approx(-np.log10(5e-8))


def test_a_vcf_that_cannot_be_plotted_says_so_without_a_figure(ui, vcf_file, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no p-values in this cohort")

    ui.set("vcf_path", str(vcf_file))
    monkeypatch.setattr("biosuite.core.ngs.manhattan_from_vcf", broken)

    ui.app._load_vcf()

    assert ui.messages_of("error") == ["no p-values in this cohort"]
    assert ui.plots == []
    assert ui.text("vcf_result").strip().startswith(f"Loaded {len(VCF_ROWS)} variants.")


# ─── Assembly ───────────────────────────────────────────────────────────────

def test_assembly_needs_a_reads_file(ui):
    ui.clear("asm_path")
    ui.clear("asm_result")

    ui.app._run_assembly()

    assert ui.messages_of("warning") == ["Select a reads file."]
    assert ui.text("asm_result").strip() == ""


def test_assembly_names_a_reads_file_that_is_not_there(ui, tmp_path):
    missing = str(tmp_path / "absent.fastq")
    expected = assemble(missing)
    assert expected.engine == "none"
    ui.set("asm_path", missing)
    ui.clear("asm_result")

    ui.app._run_assembly()

    assert ui.text("asm_result").strip() == format_assembly_report(expected).strip()
    assert f"Note: Reads file not found: {missing}" in ui.text("asm_result")
    assert ui.status() == "Assembly: 0 contigs, N50=0"
    assert ui.messages == []


def test_assembly_reports_the_contigs_it_built_from_the_reads(ui, reads_fastq):
    result = assemble(str(reads_fastq))
    assert result.num_contigs >= 1, "20 bp steps over 31-mers must overlap"
    ui.set("asm_path", str(reads_fastq))
    ui.clear("asm_result")

    ui.app._run_assembly()

    assert ui.text("asm_result").strip() == format_assembly_report(result).strip()
    assert ui.status() == f"Assembly: {result.num_contigs} contigs, N50={result.n50}"
    assert f"Contigs: {result.num_contigs}" in ui.text("asm_result")
    assert result.total_length >= len(TARGET) - 31, "the reads must stitch into sequence"
    assert ui.messages == []


def test_assembly_reports_a_reads_file_it_cannot_open(ui, tmp_path):
    directory = tmp_path / "not-a-fastq"
    directory.mkdir()
    ui.clear("asm_result")
    ui.set("asm_path", str(directory))

    ui.app._run_assembly()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "a directory is not a reads file; the tab must say so"
    assert "not-a-fastq" in errors[0], errors[0]
    assert ui.text("asm_result").strip() == ""


# ─── Metagenomics ───────────────────────────────────────────────────────────

def test_classification_needs_a_reads_file(ui):
    ui.clear("mg_path")
    ui.clear("mg_result")

    ui.app._run_meta()

    assert ui.messages_of("warning") == ["Select a reads file."]
    assert ui.text("mg_result").strip() == ""


def test_classification_reports_the_taxa_the_builtin_finder_saw(ui, taxa_fastq):
    result = classify_reads(str(taxa_fastq))
    assert len(result.classifications) == 3
    ui.set("mg_path", str(taxa_fastq))
    ui.clear("mg_result")

    ui.app._run_meta()

    assert ui.text("mg_result").strip() == format_metagenomics_report(result).strip()
    assert ui.status() == "Metagenomics: 3 reads processed"
    assert "Reads: 3" in ui.text("mg_result")
    assert "Assigned to a taxon: 2 of 3 read(s)" in ui.text("mg_result")
    assert TAXA[0] in ui.text("mg_result") and TAXA[1] in ui.text("mg_result")
    assert ui.messages == []


def test_classification_names_a_reads_file_that_is_not_there(ui, tmp_path):
    missing = str(tmp_path / "absent.fastq")
    ui.set("mg_path", missing)
    ui.clear("mg_result")

    ui.app._run_meta()

    assert ui.text("mg_result").strip() == \
        format_metagenomics_report(classify_reads(missing)).strip()
    assert f"Note: File not found: {missing}" in ui.text("mg_result")
    assert "Reads: 0" in ui.text("mg_result")
    assert ui.status() == "Metagenomics: 0 reads processed"


def test_classification_reports_a_file_it_cannot_open(ui, tmp_path):
    directory = tmp_path / "not-a-fastq-dir"
    directory.mkdir()
    ui.clear("mg_result")
    ui.set("mg_path", str(directory))

    ui.app._run_meta()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "an unreadable path must be named, not swallowed"
    assert "not-a-fastq-dir" in errors[0], errors[0]
    assert ui.text("mg_result").strip() == ""


# ─── Metagenomics diversity ─────────────────────────────────────────────────

def _abundance_csv(path, rows):
    path.write_text("taxon,count\n" + "".join(f"{t},{c}\n" for t, c in rows),
                    encoding="utf-8")
    return path


def _diversity_text(result):
    lines = [f"{result.message}\n\n"]
    for alpha in result.alpha_diversity:
        lines.append(f"{alpha['sample']}: observed={alpha['observed_taxa']}  "
                     f"shannon={alpha['shannon']}  simpson={alpha['simpson']}  "
                     f"chao1={alpha['chao1']}\n")
    if result.beta_diversity is not None:
        lines.append("\nBeta diversity (Bray-Curtis):\n")
        lines.append(result.beta_diversity.to_string())
        lines.append("\n")
    return "".join(lines).rstrip("\n")


def test_diversity_needs_paths_first(ui):
    ui.clear("mg_div_paths")
    ui.clear("mg_result")

    ui.app._run_diversity()

    assert ui.messages_of("warning") == [
        "Enter abundance CSV paths (comma-separated)."]
    assert ui.text("mg_result").strip() == ""


def test_diversity_reports_the_metrics_the_core_computed(ui, tmp_path):
    a = _abundance_csv(tmp_path / "sample_a.csv", [("sp1", 50), ("sp2", 50)])
    b = _abundance_csv(tmp_path / "sample_b.csv",
                       [("sp1", 10), ("sp2", 20), ("sp3", 30)])
    expected = analyze_diversity([pd.read_csv(a), pd.read_csv(b)])
    # Positive control: the core really measured two distinct communities.
    assert expected.alpha_diversity[0]["observed_taxa"] == 2
    assert expected.beta_diversity is not None
    assert expected.message == "Analyzed 2 samples"
    ui.set("mg_div_paths", f"{a},{b}")
    ui.clear("mg_result")

    ui.app._run_diversity()

    assert ui.text("mg_result").rstrip("\n") == _diversity_text(expected)
    assert ui.status() == "Diversity: 2 sample(s) analyzed"
    assert ui.messages == []


def test_diversity_of_one_sample_has_no_beta_matrix(ui, tmp_path):
    a = _abundance_csv(tmp_path / "only.csv", [("sp1", 7), ("sp2", 3)])
    expected = analyze_diversity([pd.read_csv(a)])
    assert expected.beta_diversity is None
    ui.set("mg_div_paths", str(a))
    ui.clear("mg_result")

    ui.app._run_diversity()

    assert ui.text("mg_result").rstrip("\n") == _diversity_text(expected)
    assert "Beta diversity" not in ui.text("mg_result")


def test_diversity_skips_blank_entries_between_paths(ui, tmp_path):
    a = _abundance_csv(tmp_path / "a.csv", [("sp1", 5)])
    b = _abundance_csv(tmp_path / "b.csv", [("sp1", 9)])
    expected = analyze_diversity([pd.read_csv(a), pd.read_csv(b)])
    ui.set("mg_div_paths", f"{a}, ,{b}")
    ui.clear("mg_result")

    ui.app._run_diversity()

    assert ui.text("mg_result").rstrip("\n") == _diversity_text(expected)
    assert ui.status() == "Diversity: 2 sample(s) analyzed"


def test_diversity_counts_a_sample_with_no_rows(ui, tmp_path):
    a = _abundance_csv(tmp_path / "full.csv", [("sp1", 5)])
    empty = tmp_path / "empty_rows.csv"
    empty.write_text("taxon,count\n", encoding="utf-8")
    expected = analyze_diversity([pd.read_csv(a), pd.read_csv(empty)])
    assert expected.message == "Analyzed 2 samples (1 with no abundance data)"
    ui.set("mg_div_paths", f"{a},{empty}")
    ui.clear("mg_result")

    ui.app._run_diversity()

    assert ui.text("mg_result").rstrip("\n") == _diversity_text(expected)
    assert "Sample_2: observed=0" in ui.text("mg_result")


def test_diversity_names_a_file_it_cannot_read(ui, tmp_path):
    missing = tmp_path / "absent.csv"
    ui.set("mg_div_paths", str(missing))
    ui.clear("mg_result")

    ui.app._run_diversity()

    errors = ui.messages_of("error")
    assert len(errors) == 1
    assert "absent.csv" in errors[0]
    assert ui.text("mg_result").strip() == ""


def test_diversity_names_an_empty_csv(ui, tmp_path):
    empty = tmp_path / "zero_bytes.csv"
    empty.write_text("", encoding="utf-8")
    ui.set("mg_div_paths", str(empty))
    ui.clear("mg_result")

    ui.app._run_diversity()

    errors = ui.messages_of("error")
    assert len(errors) == 1
    assert "zero_bytes.csv" in errors[0]
    assert ui.text("mg_result").strip() == ""


def test_diversity_names_a_table_missing_the_count_column(ui, tmp_path):
    bad = tmp_path / "wrong_columns.csv"
    bad.write_text("taxon,abundance\nsp1,5\n", encoding="utf-8")
    ui.set("mg_div_paths", str(bad))
    ui.clear("mg_result")

    ui.app._run_diversity()

    errors = ui.messages_of("error")
    assert len(errors) == 1
    assert "count" in errors[0]
    assert ui.text("mg_result").strip() == ""


# ─── 16S rRNA ───────────────────────────────────────────────────────────────

def test_16s_needs_sequences_first(ui):
    ui.clear("rna_text")
    ui.clear("rna_results")

    ui.app._run_16s()

    assert ui.messages_of("warning") == ["Enter 16S sequences first."]
    assert ui.text("rna_results").strip() == ""


def test_16s_insists_on_the_name_colon_format(ui):
    ui.set("rna_text", "just some bases with no label at all")
    ui.clear("rna_results")

    ui.app._run_16s()

    assert ui.messages_of("warning") == ["Use name:sequence format."]
    assert ui.text("rna_results").strip() == ""


def test_16s_classifies_every_labelled_line_and_ignores_the_rest(ui):
    seqs = [("Ecoli_16S", ECOLI_16S), ("Staph_16S", STAPH_16S)]
    expected = format_16s_report(classify_16s_rna(seqs))
    assert TAXA[0] in expected and TAXA[1] in expected
    ui.set("rna_text", "# a line nobody labelled\n" +
           "\n".join(f"{name}:{seq}" for name, seq in seqs))

    ui.app._run_16s()

    assert ui.text("rna_results").strip() == expected.strip()
    assert "Sequences read: 2" in ui.text("rna_results")
    assert "Ecoli_16S" not in ui.text("rna_results"), "the box names taxa, not inputs"
    assert ui.messages == []


def test_16s_reports_a_classifier_failure_without_a_stale_report(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the 16S database is empty")

    ui.set("rna_text", f"Ecoli_16S:{ECOLI_16S}")
    ui.set("rna_results", "KEEP")
    monkeypatch.setattr("biosuite.core.metagenomics.classify_16s_rna", broken)

    ui.app._run_16s()

    assert ui.messages_of("error") == ["the 16S database is empty"]
    assert ui.text("rna_results").strip() == "KEEP"


# ─── SV / CNV ───────────────────────────────────────────────────────────────

def test_a_cancelled_coverage_dialog_loads_nothing(ui):
    ui.set("svcnv_results", "KEEP")
    ui.open_path = ""

    ui.app._svcnv_load()

    assert ui.text("svcnv_results").strip() == "KEEP"
    assert ui.messages == []


def test_a_coverage_file_that_is_not_there_is_named(ui, tmp_path):
    ui.set("svcnv_results", "KEEP")
    ui.open_path = str(tmp_path / "absent.csv")

    ui.app._svcnv_load()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "absent.csv" in errors[0], errors
    assert errors[0].startswith("Could not load coverage CSV:"), errors[0]
    assert ui.text("svcnv_results").strip() == "KEEP"


def test_a_non_numeric_coverage_file_is_named(ui, tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("this column is not coverage at all\n", encoding="utf-8")
    ui.set("svcnv_results", "KEEP")
    ui.open_path = str(path)

    ui.app._svcnv_load()

    errors = ui.messages_of("error")
    assert len(errors) == 1, errors
    assert errors[0].startswith("Could not load coverage CSV:"), errors[0]
    assert "this column is not coverage" in errors[0], \
        "the message must name the value that is not coverage"
    assert ui.text("svcnv_results").strip() == "KEEP"


def test_a_coverage_csv_is_analysed_end_to_end(ui, coverage_csv):
    cov = np.loadtxt(str(coverage_csv), delimiter=",")
    ref = np.full_like(cov, float(np.median(cov)))
    assert cov.shape == (COVERAGE_POINTS,)
    ui.open_path = str(coverage_csv)

    ui.app._svcnv_load()

    text = ui.text("svcnv_results")
    assert text.startswith(f"Loaded {COVERAGE_POINTS} coverage values\n\n")
    assert format_sv_report(detect_structural_variants(cov, ref)).strip() in text
    assert format_cnv_report(detect_cnv(cov, ref)).strip() in text
    assert ui.status() == f"SV/CNV: analysed {COVERAGE_POINTS} coverage values"
    assert "DEL: " in text and "DUP: " in text, "the dip and the bump must both fire"
    assert ui.messages == []


def test_an_empty_coverage_file_is_refused(ui, tmp_path):
    """An empty file used to say "Loaded 0 coverage values" and then report a
    successful analysis of nothing."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    ui.set("svcnv_results", "KEEP")
    ui.open_path = str(path)

    ui.app._svcnv_load()

    assert ui.messages_of("error") == ["The coverage file holds no values."]
    assert ui.text("svcnv_results").strip() == "KEEP"


def test_a_non_finite_coverage_value_is_refused(ui, tmp_path):
    """A NaN makes the median NaN and the detector then reports CN=2 with a NaN
    coverage, so the file must be refused before the detectors see it."""
    path = tmp_path / "gappy.csv"
    path.write_text("30\nnan\n30\n", encoding="utf-8")
    ui.set("svcnv_results", "KEEP")
    ui.open_path = str(path)

    ui.app._svcnv_load()

    assert ui.messages_of("error") == [
        "The coverage file holds 1 missing or non-finite value(s)."]
    assert ui.text("svcnv_results").strip() == "KEEP"


def test_a_coverage_detector_failure_is_reported_after_the_load(ui, coverage_csv,
                                                               monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no bins fit the coverage")

    ui.open_path = str(coverage_csv)
    monkeypatch.setattr("biosuite.core.variant_calling.detect_cnv", broken)

    ui.app._svcnv_load()

    assert ui.messages_of("error") == ["no bins fit the coverage"]
    assert ui.text("svcnv_results").strip() == f"Loaded {COVERAGE_POINTS} coverage values"


def test_the_coverage_demo_shows_the_report_the_core_produced(ui):
    cov = np.random.default_rng(42).poisson(30, 5000).astype(float)
    cov[1500:2000] *= 0.3
    cov[3000:3500] *= 2.5
    ref = np.ones(5000) * 30
    expected = (format_sv_report(detect_structural_variants(cov, ref)) + "\n" +
                format_cnv_report(detect_cnv(cov, ref)))
    assert cov[1500:2000].mean() < 10 and cov[3000:3500].mean() > 70

    ui.clear("svcnv_results")
    ui.app._svcnv_demo()

    assert ui.text("svcnv_results").strip() == expected.strip()
    assert ui.messages == []
    assert "Total SVs:" in ui.text("svcnv_results")


def test_the_coverage_demo_leaves_the_global_rng_alone(ui):
    """The demo used ``np.random.seed(42)``, resetting the process-wide generator
    for every other tab that samples from it."""
    before = np.random.get_state()

    ui.app._svcnv_demo()

    after = np.random.get_state()
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


# ─── BigWig ─────────────────────────────────────────────────────────────────

def test_browsing_records_the_chosen_bigwig(ui, tmp_path):
    path = tmp_path / "signal.bw"
    path.write_bytes(b"not really a bigwig")
    ui.clear("bigwig_info")
    ui.open_path = str(path)

    ui.app._bigwig_browse()

    assert ui.text("bigwig_info").strip() == f"Selected: {path}"
    assert ui.app._bigwig_path == str(path)
    assert ui.messages == []


def test_a_cancelled_bigwig_browse_changes_nothing(ui, monkeypatch):
    monkeypatch.setattr(ui.app, "_bigwig_path", r"previously\chosen.bw")
    ui.set("bigwig_info", "KEEP")
    ui.open_path = ""

    ui.app._bigwig_browse()

    assert ui.text("bigwig_info").strip() == "KEEP"
    assert ui.app._bigwig_path == r"previously\chosen.bw"


def test_reading_without_a_browsed_file_asks_for_one(ui, monkeypatch):
    monkeypatch.setattr(ui.app, "_bigwig_path", None)
    ui.clear("bigwig_info")

    ui.app._bigwig_read()

    assert ui.messages_of("warning") == ["Browse for a BigWig file first."]
    assert ui.text("bigwig_info").strip() == ""


def test_a_file_that_is_not_a_bigwig_is_named(ui, tmp_path, monkeypatch):
    path = tmp_path / "junk.bw"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    monkeypatch.setattr(ui.app, "_bigwig_path", str(path))

    ui.app._bigwig_read()

    text = ui.text("bigwig_info").strip()
    assert text.startswith("Error: ") and "junk.bw" in text, text
    assert ("does not start with the BigWig magic number c74b6c27 (found 00010203), "
            "so it is not a BigWig file") in text, text
    assert ui.messages == []


def test_a_missing_bigwig_file_is_reported(ui, tmp_path, monkeypatch):
    """The reader answers with an error dict rather than raising, so the pane carries it.

    A missing file used to fall into the tab's blanket except and pop a dialog
    holding the operating system's own wording.  The message is now written where
    the user is already looking, and it names the file that was not found.
    """
    monkeypatch.setattr(ui.app, "_bigwig_path", str(tmp_path / "absent.bw"))
    ui.set("bigwig_info", "KEEP")

    ui.app._bigwig_read()

    assert ui.messages_of("error") == []
    text = ui.text("bigwig_info").strip()
    assert text.startswith("Error: ") and "absent.bw" in text, text
    assert "Could not open BigWig file" in text, text


def test_a_reader_that_raises_is_caught_by_the_tab_and_dialogued(ui, monkeypatch):
    """The tab's own boundary: anything the reader still cannot predict becomes a
    dialog, not a traceback in the console of a running session.

    The core now answers a missing file with an error dict instead of raising, so
    this stub is what keeps that handler reachable: an unexpected exception from
    an engine, a widget failure, or a malformed summary frame has to end here.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("boom in the reader")

    monkeypatch.setattr("biosuite.core.file_formats.read_bigwig", explode)
    monkeypatch.setattr(ui.app, "_bigwig_path", "signal.bw")
    ui.set("bigwig_info", "KEEP")

    ui.app._bigwig_read()

    errors = ui.messages_of("error")
    assert errors == ["boom in the reader"], errors
    assert ui.text("bigwig_info").strip() == "KEEP"


def test_a_bigwig_with_chromosomes_lists_the_values_per_chrom(ui, monkeypatch):
    """``bigWigReader`` is not installed on this machine, so the reader is stood
    in for; what is asserted is the per-chromosome list the tab itself renders."""
    chroms = {"chr1": [1.0, 2.0, 3.0], "chr2": [4.0, 5.0]}
    fake = types.ModuleType("bigWigReader")

    class _Reader:
        def __init__(self, path):
            self.path = path

        def getChroms(self):
            return list(chroms)

        def getChromSize(self, chrom):
            return len(chroms[chrom]) * 100

        def getEntries(self, chrom, start, end):
            return [(i, i + 1, value) for i, value in enumerate(chroms[chrom])]

    fake.open = lambda path: _Reader(path)
    monkeypatch.setitem(sys.modules, "bigWigReader", fake)
    monkeypatch.setattr(ui.app, "_bigwig_path", "signal.bw")

    ui.app._bigwig_read()

    assert ui.text("bigwig_info").rstrip("\n") == ("Chromosomes: 2\n\n"
                                                   "  chr1: 3 values\n  chr2: 2 values")
    assert ui.messages == []


def test_a_region_only_bigwig_falls_back_to_the_binned_summary(ui, monkeypatch, tmp_path):
    """A read without the ``chroms`` key must go through the summariser rather
    than print an empty chromosome list or a bogus error."""
    from biosuite.core.file_formats import bigwig_summary

    region = {"chrom": "chr1", "start": 0, "end": 4, "values": np.array([1.0, 2.0])}
    path = tmp_path / "signal.bw"
    path.write_bytes(b"\x00\x01\x02\x03")
    monkeypatch.setattr("biosuite.core.file_formats.read_bigwig",
                        lambda *args, **kwargs: dict(region))
    monkeypatch.setattr(ui.app, "_bigwig_path", str(path))

    ui.app._bigwig_read()

    assert ui.text("bigwig_info").strip() == format_bigwig_summary(
        bigwig_summary(str(path))).strip()
    assert "Chromosomes:" not in ui.text("bigwig_info")
    assert "Error:" not in ui.text("bigwig_info")
    assert ui.messages == []
