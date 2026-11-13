"""The Sequence, Alignment, Phylogeny and ORF-tools pages of one real app.

The sequence handlers lean on ``_run_bg``, which spawns a daemon thread; the
thread is stubbed to run inline so no test leaves one behind.  Everything
expected is recomputed from the same ``biosuite.core`` functions the tab wraps —
``sequence``, ``alignment``, ``phylogeny`` and ``orf_finder`` — so a result box
is checked against the numbers the library actually produced for the sequence
typed into it, and the input boxes are proved to reach the algorithm.
"""
import threading
import tkinter as tk

import pytest

from biosuite.core.alignment import (
    multiple_alignment,
    needleman_wunsch,
    smith_waterman,
)
from biosuite.core.orf_finder import (
    design_primers,
    find_orfs,
    find_restriction_sites,
    format_orf_results,
    format_primers,
    format_restriction_sites,
)
from biosuite.core.phylogeny import distance_matrix
from biosuite.core.sequence import (
    gc_content,
    read_fasta,
    reverse_complement,
    sequence_stats,
    translate,
)

RAW = "ATGCGGCCATGCAAAGCTTTTGACCTGA"
FASTA_RECORDS = [("alpha", "ATGCGGCCATGCAAAG"), ("beta", "TTTACGGGCCCCCA")]
GENBANK_BASES = "ATGCGGCCATGCAAACGTCGATTGCGATGCGCTTAGCTTAGCTTAGGGAATTCCTGGCAA"


class InlineThread:
    """A ``threading.Thread`` stand-in that runs its target the moment it starts.

    ``_run_bg`` hands its work to a daemon thread and marshals the callbacks back
    with ``after()``, which the session fixture already runs synchronously; with
    this stub the whole round trip happens inside the call, so no thread outlives
    the test and the results box can be asserted the instant the handler returns.
    """

    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, *, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


@pytest.fixture(autouse=True)
def inline_worker_threads(monkeypatch):
    monkeypatch.setattr(threading, "Thread", InlineThread)


@pytest.fixture
def clipboard(monkeypatch, gui_app):
    """Records clipboard traffic instead of writing over the user's real one."""
    calls = []
    monkeypatch.setattr(gui_app, "clipboard_clear",
                        lambda: calls.append(("clear", None)))
    monkeypatch.setattr(gui_app, "clipboard_append",
                        lambda text: calls.append(("append", text)))
    return calls


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "genes.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in FASTA_RECORDS),
                    encoding="utf-8")
    return path


@pytest.fixture
def fastq_file(tmp_path):
    path = tmp_path / "reads.fastq"
    path.write_text("@read1\nACGTACGTAC\n+\nIIIIIIIIII\n"
                    "@read2 length 8\nGGCCGGCC\n+\n%%%%%%%%\n", encoding="utf-8")
    return path


@pytest.fixture
def genbank_file(tmp_path):
    path = tmp_path / "plasmid.gb"
    origin = " ".join(GENBANK_BASES[i:i + 10]
                      for i in range(0, len(GENBANK_BASES), 10))
    path.write_text(
        "LOCUS       TEST001                  60 bp    DNA     linear   SYN\n"
        "DEFINITION  Test sequence for the GUI test.\n"
        "ACCESSION   TEST001\n"
        "VERSION     TEST001.1\n"
        "KEYWORDS    .\n"
        "SOURCE      synthetic construct\n"
        "  ORGANISM  synthetic construct\n"
        "            other sequences; artificial synthetic constructs.\n"
        "FEATURES             Location/Qualifiers\n"
        "     source          1..60\n"
        '                     /organism="synthetic construct"\n'
        '                     /mol_type="genomic DNA"\n'
        "ORIGIN\n"
        f"        1 {origin}\n"
        "//\n", encoding="utf-8")
    return path


@pytest.fixture
def orf_template(tmp_path):
    """A coding stretch with a stop codon at each end, so ORFs are guaranteed."""
    codons = ["ATG"] + ["GCT", "GCC", "AAA", "TTT", "GGG"] * 12 + ["TAA"]
    return "".join(codons)


# ─── Loading a sequence file ────────────────────────────────────────────────

def test_cancelling_the_file_dialog_leaves_the_sequence_box_untouched(ui):
    ui.set("seq_text", "KEEP ME")
    ui.open_path = ""

    ui.app._load_seq_file()

    assert ui.text("seq_text") == "KEEP ME\n"
    assert ui.messages == []


def test_loading_a_fasta_file_shows_every_record_it_contains(ui, fasta_file):
    ui.clear("seq_text")
    ui.open_path = str(fasta_file)

    ui.app._load_seq_file()

    expected = "\n".join(f">{name}\n{seq}" for name, seq in FASTA_RECORDS)
    assert ui.text("seq_text") == expected + "\n"
    assert ui.status().startswith("Ready"), "the loader runs off the UI thread"
    assert ui.messages == []
    assert ui.app._get_seq() == "".join(seq for _, seq in FASTA_RECORDS)


def test_loading_a_fastq_file_keeps_the_record_lines_intact(ui, fastq_file):
    ui.clear("seq_text")
    ui.open_path = str(fastq_file)

    ui.app._load_seq_file()

    assert ui.text("seq_text") == ("@read1\nACGTACGTAC\n+\nIIIIIIIIII\n"
                                   "@read2 length 8\nGGCCGGCC\n+\n%%%%%%%%\n")
    assert ui.app._get_seq() == "ACGTACGTACGGCCGGCC", "quality lines are not sequence"


def test_an_uppercase_fasta_suffix_is_read_as_fasta(ui, tmp_path):
    """The dispatch matched suffixes case-sensitively, so ``.FASTA`` fell through
    to the GenBank reader, which cannot read FASTA and returned nothing."""
    from biosuite.core.sequence import read_genbank

    path = tmp_path / "GENES.FASTA"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in FASTA_RECORDS),
                    encoding="utf-8")
    assert read_genbank(str(path)) == [], \
        "the old suffix match sent this FASTA to the GenBank reader"
    ui.clear("seq_text")
    ui.open_path = str(path)

    ui.app._load_seq_file()

    expected = "\n".join(f">{name}\n{seq}" for name, seq in FASTA_RECORDS)
    assert ui.text("seq_text") == expected + "\n"


def test_loading_a_genbank_file_shows_the_annotated_sequence(ui, genbank_file):
    from biosuite.core.sequence import read_genbank

    assert len(GENBANK_BASES) == 60
    records = read_genbank(str(genbank_file))
    assert records and records[0][1].upper() == GENBANK_BASES
    ui.clear("seq_text")
    ui.open_path = str(genbank_file)

    ui.app._load_seq_file()

    assert ui.text("seq_text").strip() == f">{records[0][0]}\n{records[0][1]}"
    assert ui.text("seq_text").strip().endswith(GENBANK_BASES)
    assert ui.messages == []


def test_a_file_that_is_not_there_loads_an_empty_box(ui, tmp_path):
    """``read_*`` answer None for a missing file; the tab must show nothing, not
    the Python ``None`` and not a stack trace."""
    for suffix in (".fasta", ".fastq", ".gb"):
        ui.set("seq_text", "stale")
        ui.open_path = str(tmp_path / f"absent{suffix}")

        ui.app._load_seq_file()

        assert ui.text("seq_text").strip() == "", f"{suffix} left text behind"
    assert ui.messages == []


def test_an_empty_fasta_file_loads_an_empty_box(ui, tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    ui.set("seq_text", "stale")
    ui.open_path = str(empty)

    ui.app._load_seq_file()

    assert read_fasta(str(empty)) == []
    assert ui.text("seq_text").strip() == ""


# ─── The live bp counter ────────────────────────────────────────────────────

def test_the_bp_counter_follows_what_is_typed(ui):
    ui.set("seq_text", f">gene\n{RAW}")

    ui.app._seq_refresh_info()

    assert ui.text("seq_info_label") == f"{len(RAW):,} bp"

    ui.set("seq_text", "   \n  ")
    ui.app._seq_refresh_info()
    assert ui.text("seq_info_label") == "0 bp", "an empty box must read as zero bases"


def test_the_bp_counter_reports_a_long_sequence_with_a_separator(ui):
    long_seq = "ACGT" * 400
    assert len(long_seq) == 1600
    ui.set("seq_text", long_seq)

    ui.app._seq_refresh_info()

    assert ui.text("seq_info_label") == "1,600 bp"


def test_the_bp_counter_stays_silent_when_its_label_cannot_be_updated(ui, monkeypatch):
    """The counter is bound to every keystroke; a half-torn-down label must not
    raise into the middle of typing."""
    attempts = []

    class BrokenLabel:
        def configure(self, **kwargs):
            attempts.append(kwargs)
            raise tk.TclError("label is gone")

    ui.set("seq_text", RAW)
    monkeypatch.setattr(ui.app, "seq_info_label", BrokenLabel())

    ui.app._seq_refresh_info()

    assert attempts == [{"text": f"{len(RAW):,} bp"}]
    assert ui.messages == []


# ─── Copying results ────────────────────────────────────────────────────────

def test_copying_an_empty_results_box_asks_for_an_analysis_first(ui, clipboard):
    ui.clear("seq_stats")

    ui.app._seq_copy_results()

    assert ui.messages == [("info", "Nothing to Copy", "Run an analysis first.")]
    assert clipboard == [], "nothing was copied, so the clipboard must be untouched"


def test_copy_sends_the_results_box_to_the_clipboard(ui, clipboard):
    ui.set("seq_stats", "line1\nline2")

    ui.app._seq_copy_results()

    assert clipboard == [("clear", None), ("append", "line1\nline2")]
    assert ui.status() == "Results copied to clipboard"
    assert ui.messages == []


# ─── GC / reverse complement / translation / stats ──────────────────────────

def test_gc_without_a_sequence_asks_for_one(ui):
    ui.clear("seq_text")
    ui.clear("seq_stats")
    before = ui.status()

    ui.app._seq_gc()

    assert ui.messages_of("warning") == ["Enter or load a sequence first."]
    assert ui.text("seq_stats").strip() == ""
    assert ui.status() == before, "the guard must return before anything is scheduled"


def test_gc_reports_the_composition_the_core_computes(ui):
    ui.set("seq_text", f">gene\n{RAW}")
    stats = sequence_stats(RAW)

    ui.app._seq_gc()

    assert ui.text("seq_stats").strip() == (
        f"Length: {len(RAW):,}\n"
        f"GC%: {gc_content(RAW):.2f}%\n"
        f"A:{stats['A']} T:{stats['T']} G:{stats['G']} C:{stats['C']} N:{stats['N']}")
    assert stats["G"] + stats["C"] == RAW.count("G") + RAW.count("C")
    assert ui.messages == []


def test_gc_names_a_sequence_that_is_not_dna(ui):
    ui.clear("seq_stats")
    ui.set("seq_text", ">junk\nATGXYZ!!!")

    ui.app._seq_gc()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "a non-nucleotide paste must be reported"
    assert "IUPAC nucleotides" in errors[0], errors[0]
    assert ui.text("seq_stats").strip() == ""


def test_reverse_complement_shows_both_strands(ui):
    ui.set("seq_text", RAW)

    ui.app._seq_revcomp()

    assert ui.text("seq_stats").strip() == (
        f"Original ({len(RAW):,} bp):\n{RAW}\n\n"
        f"Reverse Complement ({len(RAW):,} bp):\n{reverse_complement(RAW)}")
    assert "..." not in ui.text("seq_stats"), "a 26 bp sequence needs no ellipsis"
    assert ui.messages == []


def test_a_long_reverse_complement_is_shortened_with_an_ellipsis(ui):
    long_seq = "ACGT" * 70
    assert len(long_seq) == 280
    ui.set("seq_text", long_seq)

    ui.app._seq_revcomp()

    rc = reverse_complement(long_seq)
    assert ui.text("seq_stats").strip() == (
        f"Original (280 bp):\n{long_seq[:200]}...\n\n"
        f"Reverse Complement (280 bp):\n{rc[:200]}...")
    assert ui.text("seq_stats").count("...") == 2, "both strands are truncated"


def test_translation_shows_frame_one_of_the_core_code(ui):
    ui.set("seq_text", f">gene\n{RAW}")

    ui.app._seq_translate()

    protein = translate(RAW, frame=1)
    assert ui.text("seq_stats").strip() == (
        f"Translation (frame 1, {len(protein):,} aa):\n{protein}")
    assert protein.startswith("M"), "ATG must open the translation"
    assert ui.messages == []


def test_a_long_translation_is_cut_at_three_hundred_amino_acids(ui):
    coding = "ATG" + "GCT" * 400
    ui.set("seq_text", coding)

    ui.app._seq_translate()

    protein = translate(coding, frame=1)
    assert len(protein) == 401 > 300
    assert ui.text("seq_stats").strip() == (
        f"Translation (frame 1, {len(protein):,} aa):\n{protein[:300]}...")
    assert ui.text("seq_stats").count("...") == 1


def test_stats_prints_every_number_the_core_returns(ui):
    ui.set("seq_text", RAW)
    stats = sequence_stats(RAW)

    ui.app._seq_stats_cmd()

    expected = "".join(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n"
                       for k, v in stats.items())
    assert ui.text("seq_stats") == expected + "\n"
    assert f"GC: {stats['GC']:.4f}\n" in ui.text("seq_stats")
    assert f"length: {len(RAW)}\n" in ui.text("seq_stats")
    assert f"other: {stats['other']:.4f}\n" in ui.text("seq_stats")
    assert ui.messages == []


def test_a_fresh_run_replaces_what_the_results_box_held(ui):
    """``done()`` clears the box first: a stale report must not survive a new run."""
    ui.set("seq_stats", "STALE REPORT FROM THE LAST RUN")
    ui.set("seq_text", "ATGC")

    ui.app._seq_stats_cmd()

    assert "STALE" not in ui.text("seq_stats")
    assert "GC: 50.0000\n" in ui.text("seq_stats")
    assert sequence_stats("ATGC")["GC"] == 50.0


# ─── Pairwise alignment ─────────────────────────────────────────────────────

def test_alignment_refuses_to_run_without_both_sequences(ui):
    ui.set("align_seq1", "ACGT")
    ui.clear("align_seq2")
    ui.clear("align_result")

    ui.app._align_nw()
    ui.app._align_sw()

    assert ui.messages_of("warning") == ["Enter both sequences.",
                                         "Enter both sequences."]
    assert ui.text("align_result").strip() == ""


def test_alignment_refuses_when_the_first_sequence_is_the_missing_one(ui):
    ui.clear("align_seq1")
    ui.set("align_seq2", "ACGT")
    ui.clear("align_result")

    ui.app._align_nw()

    assert ui.messages_of("warning") == ["Enter both sequences."]
    assert ui.text("align_result").strip() == ""


def test_needleman_wunsch_writes_the_score_and_the_pair(ui):
    s1, s2 = "ACGTACGT", "ACGTAGT"
    a1, a2, score = needleman_wunsch(s1, s2)
    ui.set("align_seq1", s1)
    ui.set("align_seq2", s2)

    ui.app._align_nw()

    assert ui.text("align_result").strip() == f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}"
    assert a1.replace("-", "") == s1 and a2.replace("-", "") == s2, "global aligns all of both"
    scored = sum(1 if x == y else (-2 if "-" in (x, y) else -1) for x, y in zip(a1, a2))
    assert scored == score, "the score shown must be the score of the pair shown"
    assert ui.messages == []


def test_smith_waterman_writes_the_local_pair(ui):
    s1 = "TTTTTTTTACGTACGT"
    s2 = "ACGTACGTGGGG"
    a1, a2, score = smith_waterman(s1, s2)
    ui.set("align_seq1", s1)
    ui.set("align_seq2", s2)

    ui.app._align_sw()

    assert ui.text("align_result").strip() == f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}"
    assert a1.replace("-", "") == "ACGTACGT", "local alignment must ignore the flanks"
    assert score == 8


@pytest.mark.parametrize("handler,function", [
    ("_align_nw", "needleman_wunsch"),
    ("_align_sw", "smith_waterman"),
])
def test_an_alignment_failure_is_reported_instead_of_escaping(ui, monkeypatch,
                                                              handler, function):
    """Neither alignment handler had a boundary around the core call, so a
    failure would print a traceback to stderr and leave the box untouched."""
    def broken(*args, **kwargs):
        raise RuntimeError("alignment engine unavailable")

    monkeypatch.setattr(f"biosuite.core.alignment.{function}", broken)
    ui.set("align_seq1", "ACGT")
    ui.set("align_seq2", "ACGT")
    ui.set("align_result", "KEEP THIS REPORT")

    getattr(ui.app, handler)()

    assert ui.messages_of("error") == ["alignment engine unavailable"]
    assert ui.text("align_result").strip() == "KEEP THIS REPORT"


def test_the_alignment_ignores_spaces_and_line_breaks(ui):
    s1, s2 = "ACGTACGT", "ACGTAGT"
    a1, a2, score = needleman_wunsch(s1, s2)
    ui.set("align_seq1", "ACGT ACGT\n")
    ui.set("align_seq2", "ACG TAG T\n")

    ui.app._align_nw()

    assert ui.text("align_result").strip() == f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}"


# ─── Multiple sequence alignment ────────────────────────────────────────────

MSA_NAMES = ["seqA", "seqB", "seqC"]
MSA_SEQS = ["ACGTACGT", "ACGTAGT", "ACGTTACGT"]


def _msa_fasta(names=None, seqs=None):
    names = MSA_NAMES if names is None else names
    seqs = MSA_SEQS if seqs is None else seqs
    return "\n".join(f">{name}\n{seq}" for name, seq in zip(names, seqs))


def _msa_text(names, aligned):
    lines = [f"Multiple alignment ({len(aligned)} sequences):\n\n"]
    lines.extend(f"{name}: {row}\n" for name, row in zip(names, aligned))
    return "".join(lines).rstrip("\n")


def test_msa_without_data_asks_for_some(ui):
    ui.clear("align_msa_input")
    ui.clear("align_result")

    ui.app._align_msa()

    assert ui.messages_of("warning") == [
        "Enter sequences in FASTA format (>name then sequence)."]
    assert ui.text("align_result").strip() == ""


def test_the_msa_report_matches_the_core_alignment(ui):
    aligned = multiple_alignment(MSA_SEQS)
    # Positive control: the core aligned all three rows to one common length.
    assert len({len(row) for row in aligned}) == 1
    assert len(aligned[0]) >= max(len(s) for s in MSA_SEQS)
    ui.set("align_msa_input", _msa_fasta())
    ui.clear("align_result")

    ui.app._align_msa()

    assert ui.text("align_result").rstrip("\n") == _msa_text(MSA_NAMES, aligned)
    assert ui.status() == "MSA: 3 sequences aligned"
    assert ui.messages == []


def test_msa_needs_two_sequences(ui):
    ui.set("align_msa_input", ">only\nACGTACGT\n")
    ui.set("align_result", "KEEP THIS REPORT")

    ui.app._align_msa()

    assert ui.messages_of("error") == [
        "Need at least 2 sequences for multiple alignment."]
    assert ui.text("align_result").strip() == "KEEP THIS REPORT"


def test_msa_ignores_blank_lines_between_records(ui):
    aligned = multiple_alignment(MSA_SEQS[:2])
    ui.set("align_msa_input", ">seqA\nACGTACGT\n\n>seqB\nACGTAGT\n")
    ui.clear("align_result")

    ui.app._align_msa()

    assert ui.text("align_result").rstrip("\n") == _msa_text(MSA_NAMES[:2], aligned)
    assert ui.messages == []


def test_msa_names_a_record_without_a_sequence(ui):
    ui.set("align_msa_input", ">a\nACGT\n>b\nTTTT\n>c\n")
    ui.set("align_result", "KEEP THIS REPORT")

    ui.app._align_msa()

    assert ui.messages_of("error") == ["Every sequence needs its own name line."]
    assert ui.text("align_result").strip() == "KEEP THIS REPORT"


def test_an_msa_failure_is_reported_without_touching_the_old_report(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("msa engine unavailable")
    monkeypatch.setattr("biosuite.core.alignment.multiple_alignment", broken)
    ui.set("align_msa_input", _msa_fasta())
    ui.set("align_result", "KEEP THIS REPORT")

    ui.app._align_msa()

    assert ui.messages_of("error") == ["msa engine unavailable"]
    assert ui.text("align_result").strip() == "KEEP THIS REPORT"


# ─── Phylogenetic tree ──────────────────────────────────────────────────────

TREE_NAMES = ["speciesA", "speciesB", "speciesC"]
TREE_SEQS = ["ACGTACGTAC", "ACGTACGTAG", "ACGTACGTAA"]


def _tree_fasta():
    return "\n".join(f">{name}\n{seq}" for name, seq in zip(TREE_NAMES, TREE_SEQS))


def test_the_tree_builder_needs_data_first(ui):
    ui.clear("phylo_input")
    ui.clear("phylo_result")

    ui.app._build_tree()

    assert ui.messages_of("warning") == [
        "Enter aligned sequences (>name and sequence)."]
    assert ui.text("phylo_result").strip() == ""


def test_one_sequence_cannot_make_a_tree(ui):
    ui.set("phylo_input", ">only\nACGTACGT")
    ui.clear("phylo_result")

    ui.app._build_tree()

    assert ui.messages_of("error") == ["Need at least 2 sequences."]
    assert ui.plots == []


def test_unaligned_input_is_refused_before_any_matrix_is_built(ui):
    ui.set("phylo_input", ">a\nACGTACGT\n>b\nACGT")
    ui.clear("phylo_result")

    ui.app._build_tree()

    assert ui.messages_of("error") == ["Sequences must be same length (aligned)."]
    assert ui.plots == []


def test_the_tree_lists_the_distance_matrix_and_shows_the_figure(ui):
    dist = distance_matrix(TREE_SEQS)
    expected_rows = "".join(
        f"{name}: " + ", ".join(f"{dist[i][j]:.3f}" for j in range(3)) + "\n"
        for i, name in enumerate(TREE_NAMES))
    ui.set("phylo_input", _tree_fasta())

    ui.app._build_tree()

    assert ui.text("phylo_result").strip() == (
        "Distance Matrix:\n" + expected_rows + "\nTree displayed in matplotlib window.")
    assert dist[0][1] == pytest.approx(0.1), "the last base differs in one of ten sites"
    assert ui.messages == []
    title, fig = ui.plots[-1]
    assert title == "Phylogenetic Tree"
    assert sorted(label.get_text() for label in fig.axes[0].get_xticklabels()) == TREE_NAMES


def test_a_record_without_a_sequence_is_reported_not_plotted(ui):
    """The parser only counts a record once sequence lines follow the header."""
    ui.set("phylo_input", ">alpha\nACGTACGTAC\n>beta\n")
    ui.clear("phylo_result")

    ui.app._build_tree()

    assert ui.messages_of("error") == ["Need at least 2 sequences."]
    assert ui.plots == []
    assert ui.text("phylo_result").strip() == ""


def test_more_names_than_sequences_is_refused_before_anything_is_drawn(ui):
    """A trailing '>name' with no sequence leaves two records and three names; the
    matrix rows used to be written from the names and then run off the end of the
    distance matrix, over a half-filled box and an already-displayed figure."""
    ui.set("phylo_input", ">alpha\nACGTACGTAC\n>beta\nACGTACGTAC\n>gamma")
    ui.clear("phylo_result")

    ui.app._build_tree()

    assert ui.messages_of("error") == ["Every sequence needs its own name line."]
    assert ui.plots == []
    assert ui.text("phylo_result").strip() == ""


def test_a_tree_failure_is_reported_instead_of_a_stale_plot(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("scipy cannot cluster these")

    ui.set("phylo_input", _tree_fasta())
    ui.set("phylo_result", "KEEP")
    monkeypatch.setattr("biosuite.core.phylogeny.plot_phylogenetic_tree", broken)

    ui.app._build_tree()

    assert ui.messages_of("error") == ["scipy cannot cluster these"]
    assert ui.plots == [], "a failed tree must not display a figure"
    assert ui.text("phylo_result") == "KEEP\n"


# ─── ORF finder / restriction map / primer design ───────────────────────────

def test_orf_finding_needs_a_sequence(ui):
    ui.clear("orf_seq")
    ui.clear("orf_result")

    ui.app._run_orf()

    assert ui.messages_of("warning") == ["Enter a DNA sequence."]
    assert ui.text("orf_result").strip() == ""


def test_the_orf_report_matches_the_core_finder(ui, orf_template):
    orfs = find_orfs(orf_template.upper())
    assert orfs, "the template is built to contain an ORF"
    ui.set("orf_seq", orf_template)

    ui.app._run_orf()

    assert ui.text("orf_result").strip() == format_orf_results(orfs).strip()
    assert f"Found {len(orfs)} ORFs" in ui.text("orf_result")
    assert ui.status() == f"Found {len(orfs)} ORFs"
    assert orfs[0].length == len(orfs[0].protein)
    assert "Frame" in ui.text("orf_result"), "the report is a table, not a blob"
    assert ui.messages == []


def test_an_orf_failure_is_reported(ui, monkeypatch, orf_template):
    def broken(*args, **kwargs):
        raise RuntimeError("no reading frames")

    ui.set("orf_seq", orf_template)
    ui.set("orf_result", "KEEP")
    monkeypatch.setattr("biosuite.core.orf_finder.find_orfs", broken)

    ui.app._run_orf()

    assert ui.messages_of("error") == ["no reading frames"]
    assert ui.text("orf_result") == "KEEP\n"


def test_restriction_mapping_needs_a_sequence(ui):
    ui.clear("orf_seq")
    ui.clear("orf_result")

    ui.app._run_restriction()

    assert ui.messages_of("warning") == ["Enter a DNA sequence."]
    assert ui.text("orf_result").strip() == ""


def test_the_restriction_report_lists_the_sites_the_core_found(ui):
    target = "GAATTCGGGAAAATTTCCCGGGTCTAGA"
    sites = find_restriction_sites(target.upper())
    assert sites, "EcoRI, SmaI and XbaI sites are in this string"
    ui.set("orf_seq", target)

    ui.app._run_restriction()

    assert ui.text("orf_result").strip() == format_restriction_sites(sites).strip()
    assert f"Found {len(sites)} cut sites" in ui.text("orf_result")
    assert ui.status() == f"Found {len(sites)} restriction sites"
    assert any(s.enzyme == "EcoRI" for s in sites)
    assert ui.messages == []


def test_a_restriction_failure_is_reported(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("enzyme table is empty")

    ui.set("orf_seq", "GAATTCGGGAAAATTTCCCGGG")
    ui.set("orf_result", "KEEP")
    monkeypatch.setattr("biosuite.core.orf_finder.find_restriction_sites", broken)

    ui.app._run_restriction()

    assert ui.messages_of("error") == ["enzyme table is empty"]
    assert ui.text("orf_result") == "KEEP\n"


def test_primer_design_needs_a_template(ui):
    ui.clear("orf_seq")
    ui.clear("orf_result")

    ui.app._run_primers()

    assert ui.messages_of("warning") == ["Enter a template DNA sequence."]
    assert ui.text("orf_result").strip() == ""


def test_the_primer_report_matches_the_core_designer(ui):
    template = ("ATGGCTAAAATCGATTGCATGGCCATGCATTTGCAACTTGGCCCTTTGCAAGCTTTGCG"
                "ATGCCTAAAGGCCCAAACTGCAAGGCTTTACAGCCATGGATCAAGGCTCCTTTGCAACTTGGCC")
    fwd, rev = design_primers(template.upper())
    assert fwd and rev, "a 120 bp template must yield both flanks"
    ui.set("orf_seq", template)

    ui.app._run_primers()

    assert ui.text("orf_result").strip() == format_primers(fwd, rev).strip()
    assert f"Forward: {fwd.sequence}" in ui.text("orf_result")
    assert f"Reverse: {rev.sequence}" in ui.text("orf_result")
    assert ui.messages == []


def test_a_primer_failure_is_reported(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no flank clears the melting temperature")

    ui.set("orf_seq", "ATGGCTAAAATCGATTGCATGGCCATGCATTTGCAACTTGGCC")
    ui.set("orf_result", "KEEP")
    monkeypatch.setattr("biosuite.core.orf_finder.design_primers", broken)

    ui.app._run_primers()

    assert ui.messages_of("error") == ["no flank clears the melting temperature"]
    assert ui.text("orf_result") == "KEEP\n"


def test_a_template_too_short_to_flank_shows_only_the_report_header(ui):
    """A positive control for the primer test above: with no flank the core
    returns two ``None``s, and the box must say so instead of inventing primers."""
    fwd, rev = design_primers("ATGCGT")
    assert fwd is None and rev is None, "a 6 bp template cannot carry 20 bp primers"
    ui.set("orf_seq", "ATGCGT")
    ui.clear("orf_result")

    ui.app._run_primers()

    assert ui.text("orf_result").strip() == "=== Designed Primers ==="
    assert "Forward" not in ui.text("orf_result")
    assert ui.messages == []
