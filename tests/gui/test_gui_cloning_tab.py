"""The Molecular Cloning tab: sequence entry, enzyme table, digest, gel, PCR.

Every handler reads the entries, calls ``biosuite.core.cloning`` (or
``core.utils`` / ``core.sequence``) through ``_run_bg``, and writes the report
into the results box.  ``_run_bg``'s worker thread is run inline so a test can
assert the box the instant the handler returns, and every expected report here is
recomputed by calling the same core function the tab wraps on the same string the
test typed — so the box is checked against what the library actually produced for
that input, never against a remembered string.
"""
import threading
import tkinter as tk
import types

import matplotlib.pyplot as plt
import pytest

from biosuite.core import utils as core_utils
from biosuite.core.cloning import (
    _VALID_DNA,
    design_primers,
    format_digest_report,
    format_pcr_report,
    format_primer_report,
    simulate_digestion,
    simulate_pcr,
    verify_insert,
    _reverse_complement,
)
from biosuite.core.sequence import read_fasta
from biosuite.gui import main_window

RECORD_ONE = ("pTRACER vector backbone", "GAATTCGGATCCACGTGCGGCCGCCTCTAGAGGATCC")
RECORD_TWO = ("junk insert", "TTTTTTTTGGATCCAAAAAAAA")
SITE = "GAATTC"
PLASMID = ("GAATTC" + "CCCCCCCCCCCCCCCC" ) * 2 + "GGATCC" + "ATATATATATATATATATAT"
FWD = "ACGTACGTAA"
REV_RC = "TTTGGGCCCAAAGG"
REV = _reverse_complement(REV_RC)
PCR_TEMPLATE = "CCCCCC" + FWD + "ATATATATATATATATATAT" + REV_RC + "GGGGGG"
PRIMER_TARGET = ("ATGGCTAAAATCGATTGCATGGCCATGCATTTGCAACTTGGCCCTTTGCAAGCTTTGCG"
                 "ATGCCTAAAGGCCCAAACTGCAAGGCTTTACAGCCATGGATCAAGGCTCC")


class _Broken:
    """A stand-in widget whose every ``configure`` fails, like a half-torn-down label."""

    def configure(self, **kwargs):
        raise tk.TclError("widget is gone")


@pytest.fixture(autouse=True)
def inline_worker_threads(monkeypatch):
    """Replace ``main_window``'s ``threading`` with a stub that runs the target inline."""
    class _Inline(threading.Thread):
        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(main_window, "threading",
                        types.SimpleNamespace(Thread=_Inline, Event=threading.Event,
                                              Lock=threading.Lock))


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
def status_log(ui, monkeypatch):
    """Every status line the handler pushes, in order, while the real bar still updates."""
    seen = []
    real = ui.app._set_status

    def spy(text, elapsed=None):
        seen.append(text)
        return real(text, elapsed=elapsed)

    monkeypatch.setattr(ui.app, "_set_status", spy)
    return seen


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "plasmid.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in (RECORD_ONE, RECORD_TWO)),
                    encoding="utf-8")
    return path


@pytest.fixture
def genbank_file(tmp_path):
    path = tmp_path / "plasmid.gb"
    path.write_text(
        "LOCUS       TEST001                  12 bp    DNA     linear   SYN\n"
        "DEFINITION  Test sequence for the GUI test.\n"
        "ACCESSION   TEST001\n"
        "VERSION     TEST001.1\n"
        "KEYWORDS    .\n"
        "SOURCE      synthetic construct\n"
        "  ORGANISM  synthetic construct\n"
        "FEATURES             Location/Qualifiers\n"
        "     source          1..12\n"
        "ORIGIN\n"
        "        1 atgcggccatgc\n"
        "//\n", encoding="utf-8")
    return path


@pytest.fixture
def digest_ready(ui):
    """The digestion inputs set from scratch, because the app is shared all session."""
    ui.app.cloning_enzyme.set("EcoRI")
    ui.app.cloning_topo.set("circular")
    ui.set("cloning_seq", PLASMID)
    ui.clear("cloning_result")
    return ui


# ─── The enzyme database ────────────────────────────────────────────────────

def test_the_enzyme_list_is_the_core_database_sorted(ui):
    expected = sorted(core_utils.RESTRICTION_ENZYMES_SITES.keys())
    assert len(expected) == len(core_utils.RESTRICTION_ENZYMES) > 100

    assert ui.app._cloning_enzyme_names() == expected
    assert "EcoRI" in expected and expected == sorted(expected)


def test_the_enzyme_list_falls_back_when_the_database_import_fails(ui, monkeypatch):
    """``from ...core.utils import RESTRICTION_ENZYMES_SITES`` must not raise into
    the middle of building the tab; the combo still has to offer real enzymes."""
    monkeypatch.delattr(core_utils, "RESTRICTION_ENZYMES_SITES")

    names = ui.app._cloning_enzyme_names()

    assert names
    assert set(names) <= set(core_utils.RESTRICTION_ENZYMES), "the fallback names a non-existent enzyme"
    assert len(names) < len(core_utils.RESTRICTION_ENZYMES), "the fallback is not a subset of the real table"
    assert "EcoRI" in names


def test_the_enzyme_line_shows_the_core_site_and_cut_offset(ui):
    ui.app.cloning_enzyme.set("EcoRI")
    site, offset = core_utils.RESTRICTION_ENZYMES["EcoRI"]
    assert site == core_utils.RESTRICTION_ENZYMES_SITES["EcoRI"]

    ui.app._cloning_refresh_enzyme_info()

    assert ui.text("cloning_enzyme_info") == (
        f"site: {site}   ·   cut offset: +{offset}   ·   palindromic site")


def test_an_even_length_site_that_is_not_a_palindrome_is_not_labelled(ui):
    """Length parity is not symmetry: BsaI's GGTCTC is six bases and its reverse
    complement is GAGACC."""
    from biosuite.core.sequence import reverse_complement

    name = "BsaI"
    site, offset = core_utils.RESTRICTION_ENZYMES[name]
    assert len(site) % 2 == 0 and reverse_complement(site) != site
    ui.app.cloning_enzyme.set(name)

    ui.app._cloning_refresh_enzyme_info()

    shown = ui.text("cloning_enzyme_info")
    assert shown == f"site: {site}   ·   cut offset: +{offset}", shown


def test_an_odd_length_site_can_still_be_a_palindrome(ui):
    """The old parity check also hid a real palindrome: BstNI's CCWGG is five
    bases and equals its own reverse complement."""
    from biosuite.core.sequence import reverse_complement

    name = "BstNI"
    site, offset = core_utils.RESTRICTION_ENZYMES[name]
    assert len(site) % 2 == 1 and reverse_complement(site) == site
    ui.app.cloning_enzyme.set(name)

    ui.app._cloning_refresh_enzyme_info()

    assert ui.text("cloning_enzyme_info") == (
        f"site: {site}   ·   cut offset: +{offset}   ·   palindromic site")


def test_an_unknown_enzyme_is_named_with_the_database_size(ui):
    ui.app.cloning_enzyme.set("NotAnEnzyme")
    before = len(core_utils.RESTRICTION_ENZYMES_SITES)

    ui.app._cloning_refresh_enzyme_info()

    assert ui.text("cloning_enzyme_info") == (
        f"'NotAnEnzyme' not in database — {before} enzymes available")
    assert ui.messages == []


def test_a_database_entry_without_a_cut_offset_shows_only_the_site(ui, monkeypatch):
    """The tab guards a malformed ``(site, offset)`` record; without the offset the
    line must simply omit it rather than print ``cut offset: +?``."""
    monkeypatch.setattr(core_utils, "RESTRICTION_ENZYMES", {"EcoRI": ("GAATTC",)})
    ui.app.cloning_enzyme.set("EcoRI")

    ui.app._cloning_refresh_enzyme_info()

    shown = ui.text("cloning_enzyme_info")
    assert shown == "site: GAATTC   ·   palindromic site", shown
    assert "cut offset" not in shown


def test_the_enzyme_line_survives_a_label_that_cannot_be_updated(ui, monkeypatch):
    """The line is refreshed from the combobox callback; a half-torn-down label
    must not raise into the selection."""
    ui.app.cloning_enzyme.set("EcoRI")
    monkeypatch.setattr(ui.app, "cloning_enzyme_info", _Broken())

    ui.app._cloning_refresh_enzyme_info()

    assert ui.messages == []


def test_a_broken_enzyme_table_names_the_failure_in_the_info_line(ui, monkeypatch):
    class _Boom:
        def get(self, name):
            raise RuntimeError("table is corrupt")

    ui.app.cloning_enzyme.set("EcoRI")
    ui.app._cloning_refresh_enzyme_info()
    assert ui.text("cloning_enzyme_info"), "the enzyme is in the real table"

    monkeypatch.setattr(core_utils, "RESTRICTION_ENZYMES_SITES", _Boom())
    ui.app._cloning_refresh_enzyme_info()

    assert ui.text("cloning_enzyme_info") == \
        "enzyme information unavailable: table is corrupt"
    assert ui.messages == []


# ─── The sequence entry, its counter and its validator ──────────────────────

def test_cleaning_the_entry_uppercases_and_strips_whitespace(ui):
    ui.set("cloning_seq", "  gaattc\ncg  gg  ")

    cleaned = ui.app._cloning_clean_seq()

    assert cleaned == "GAATTCCGGG"
    assert not any(c.isspace() for c in cleaned)
    assert cleaned == cleaned.upper()


def test_the_bp_counter_tracks_the_cleaned_sequence(ui):
    ui.clear("cloning_seq")
    ui.app._cloning_refresh_seq_info()
    assert ui.text("cloning_seq_info") == "0 bp"

    spaced = "acgt acgt\nacgtn "
    ui.set("cloning_seq", spaced)
    ui.app._cloning_refresh_seq_info()

    cleaned = ui.app._cloning_clean_seq()
    assert cleaned == "ACGTACGTACGTN"
    assert ui.text("cloning_seq_info") == f"{len(cleaned):,} bp"
    assert ui.text("cloning_seq_warn") == ""


def test_the_bp_counter_reports_a_long_sequence_with_a_separator(ui):
    seq = "ACGT" * 400
    assert len(seq) == 1600
    ui.set("cloning_seq", seq)

    ui.app._cloning_refresh_seq_info()

    assert ui.text("cloning_seq_info") == f"{len(seq):,} bp"
    assert ui.text("cloning_seq_info") == "1,600 bp"


def test_the_warning_names_exactly_the_characters_the_core_would_reject(ui):
    junk = "ACGTXQ9!"
    ui.set("cloning_seq", junk)

    ui.app._cloning_refresh_seq_info()

    cleaned = ui.app._cloning_clean_seq()
    bad = sorted(set(cleaned) - _VALID_DNA)
    assert bad == sorted("XQ9!"), "the tab and the core must disagree about nothing"
    assert ui.text("cloning_seq_warn") == f"⚠ invalid characters: {' '.join(bad)}"
    with pytest.raises(ValueError, match="Invalid characters"):
        simulate_digestion(cleaned, "EcoRI")


def test_the_warning_shows_at_most_six_bad_characters(ui):
    seq = "ACGT" + "0123456789!@#$"
    ui.set("cloning_seq", seq)

    ui.app._cloning_refresh_seq_info()

    bad = sorted(set(ui.app._cloning_clean_seq()) - _VALID_DNA)
    assert len(bad) == 14 > 6, "the input must carry more offenders than the line can show"
    shown = ui.text("cloning_seq_warn")
    assert shown == "⚠ invalid characters: " + " ".join(bad[:6])
    assert bad[6] not in shown


def test_the_counter_survives_a_label_that_cannot_be_updated(ui, monkeypatch):
    """It is bound to every keystroke, so a failing label must not raise into the entry."""
    ui.set("cloning_seq", PLASMID)
    monkeypatch.setattr(ui.app, "cloning_seq_warn", _Broken())

    ui.app._cloning_refresh_seq_info()

    assert ui.text("cloning_seq_info") == f"{len(PLASMID):,} bp"
    assert ui.messages == []


# ─── Copying the report ─────────────────────────────────────────────────────

def test_copying_an_empty_results_box_asks_for_an_analysis_first(ui, clipboard):
    ui.clear("cloning_result")

    ui.app._cloning_copy_results()

    assert ui.messages == [("info", "Nothing to Copy", "Run an analysis first.")]
    assert clipboard == [], "an empty box must leave the clipboard alone"


def test_copy_sends_the_results_box_to_the_clipboard(ui, clipboard):
    report = "=== DIGEST ===\nfragment 1\nfragment 2"
    ui.set("cloning_result", report)

    ui.app._cloning_copy_results()

    assert clipboard == [("clear", None), ("append", report)]
    assert ui.status() == "Results copied to clipboard"
    assert ui.messages == []


# ─── Loading a plasmid from file ────────────────────────────────────────────

def test_cancelling_the_file_dialog_leaves_the_sequence_untouched(ui):
    ui.set("cloning_seq", "KEEPME")
    ui.open_path = ""

    ui.app._cloning_load_file()

    assert ui.text("cloning_seq") == "KEEPME"
    assert ui.messages == []


def test_loading_a_fasta_file_puts_the_first_record_in_the_entry(ui, fasta_file, status_log):
    records = read_fasta(str(fasta_file))
    assert len(records) == 2, "the file must hold two records for this to prove anything"
    ui.clear("cloning_seq")
    ui.open_path = str(fasta_file)

    ui.app._cloning_load_file()

    assert ui.text("cloning_seq") == records[0][1]
    assert ui.messages_of("warning") == [
        f"Loaded '{records[0][0]}' (record 1 of 2)."]
    assert status_log[0] == f"Loading {fasta_file.name}..."
    assert f"Loaded {fasta_file.name} — {len(records[0][1]):,} bp" in status_log
    assert ui.text("cloning_seq_info") == f"{len(records[0][1]):,} bp"


def test_loading_a_genbank_file_reads_its_sequence(ui, genbank_file):
    """The dialog offered GenBank, but every chosen file went through
    ``read_fasta``; a GenBank file has no ``>`` records and loaded nothing."""
    from biosuite.core.sequence import read_genbank

    records = read_genbank(str(genbank_file))
    assert records and records[0][1].upper() == "ATGCGGCCATGC"
    ui.clear("cloning_seq")
    ui.open_path = str(genbank_file)

    ui.app._cloning_load_file()

    assert ui.text("cloning_seq").upper() == "ATGCGGCCATGC"
    assert ui.messages == []


def test_loading_a_file_without_records_is_reported(ui, tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    assert read_fasta(str(empty)) == []
    ui.set("cloning_seq", "KEEPME")
    ui.open_path = str(empty)

    ui.app._cloning_load_file()

    assert ui.messages_of("error") == ["No sequences found in file"]
    assert ui.text("cloning_seq") == "KEEPME"


def test_a_sequence_beyond_the_gui_limit_is_truncated_with_a_warning(ui, tmp_path, status_log):
    limit = 200_000
    body = "ACGTT" * (limit // 5 + 3)
    path = tmp_path / "huge.fasta"
    path.write_text(f">huge\n{body}\n", encoding="utf-8")
    assert read_fasta(str(path)) == [("huge", body)], "the file must hold the whole over-long record"
    ui.clear("cloning_seq")
    ui.open_path = str(path)

    ui.app._cloning_load_file()

    assert len(ui.text("cloning_seq")) == limit < len(body)
    assert ui.text("cloning_seq") == body[:limit]
    assert ui.messages_of("warning") == [
        f"Very long input — only the first {limit:,} bp were loaded\n"
        "(GUI limit to keep the entry responsive)."]
    assert f"Loaded huge.fasta — {limit:,} bp" in status_log
    assert ui.text("cloning_seq_info") == f"{limit:,} bp"


# ─── Restriction digestion ──────────────────────────────────────────────────

def test_digestion_writes_the_report_the_core_produces(digest_ready, status_log):
    expected = simulate_digestion(PLASMID, "EcoRI", topology="circular")
    assert len(expected["cuts"]) == 2, "the template must actually get cut"

    ui = digest_ready
    ui.app._run_digestion()

    assert ui.text("cloning_result").strip() == format_digest_report(expected).strip()
    assert ui.app._last_digest_result == expected
    assert ui.messages == []
    assert status_log[0] == "Running EcoRI digestion..."
    assert (f"Digestion complete: {len(expected['fragments'])} fragments, "
            f"{len(expected['cuts'])} cuts") in status_log


def test_the_enzyme_and_topology_choices_reach_the_digest(ui):
    circular = simulate_digestion(PLASMID, "BamHI", topology="circular")
    linear = simulate_digestion(PLASMID, "BamHI", topology="linear")
    assert circular["sizes"] != linear["sizes"], "the two topologies must differ on this sequence"
    ui.app.cloning_enzyme.set("BamHI")
    ui.app.cloning_topo.set("linear")
    ui.set("cloning_seq", PLASMID.lower() + "   \t")
    ui.clear("cloning_result")

    ui.app._run_digestion()

    assert ui.messages == []
    assert ui.text("cloning_result").strip() == format_digest_report(linear).strip()
    assert ui.app._last_digest_result == linear
    assert ui.app._last_digest_result != circular


def test_digestion_without_a_sequence_asks_for_one(ui, status_log):
    ui.clear("cloning_seq")
    ui.clear("cloning_result")
    before = list(status_log)

    ui.app._run_digestion()

    assert ui.messages == [("warning", "Input Required", "Please enter or load a DNA sequence.")]
    assert ui.text("cloning_result").strip() == ""
    assert status_log == before, "the guard must return before any work is scheduled"


def test_digestion_reports_an_unknown_enzyme_and_keeps_the_previous_report(ui, status_log):
    ui.set("cloning_result", "KEEP THIS REPORT")
    ui.set("cloning_seq", PLASMID)
    ui.app.cloning_enzyme.set("NotAnEnzyme")

    ui.app._run_digestion()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and errors[0].startswith("Unknown enzyme: NotAnEnzyme"), errors
    assert ui.text("cloning_result").strip() == "KEEP THIS REPORT"
    assert status_log[0] == "Running NotAnEnzyme digestion..."
    assert not any("Digestion complete" in line for line in status_log)


# ─── Virtual gel ────────────────────────────────────────────────────────────

def test_virtual_gel_without_a_sequence_asks_for_one(ui):
    plt.close("all")
    ui.clear("cloning_seq")

    ui.app._run_virtual_gel()

    assert ui.messages_of("warning") == ["Please enter or load a DNA sequence."]
    assert ui.plots == []


def test_the_virtual_gel_records_the_figure_the_core_drew(digest_ready):
    plt.close("all")
    ui = digest_ready
    expected = simulate_digestion(PLASMID, "EcoRI", topology="circular")

    ui.app._run_virtual_gel()

    assert ui.plots, "the tab never handed its figure to the viewer"
    title, fig = ui.plots[-1]
    assert title == "Cloning Plot"
    assert fig is ui.app._plot_history[-1]["fig"]
    assert ui.app._plot_history[-1]["name"] == "Virtual Gel (EcoRI)"
    assert fig.axes[0].get_title() == "EcoRI digest (circular)"
    assert ui.app._last_digest_result == expected
    assert ui.messages == []
    assert ui.status().startswith("Ready")


def test_the_gel_shows_a_band_for_every_fragment_of_the_digest(digest_ready):
    plt.close("all")
    ui = digest_ready
    ui.app.cloning_topo.set("linear")
    expected = simulate_digestion(PLASMID, "EcoRI", topology="linear")
    assert expected["sizes"], "a digest with no fragments would prove nothing"

    ui.app._run_virtual_gel()

    fig = ui.plots[-1][1]
    bands = [t.get_text() for t in fig.axes[0].texts if t.get_text().endswith(" bp")]
    assert bands == [f"{size} bp" for size in expected["sizes"]], bands
    assert len(bands) == len(expected["fragments"])


def test_an_enzyme_that_leaves_no_fragments_is_announced(ui, monkeypatch):
    """The core never returns an empty lane for a non-empty template, so the
    ``if not result['sizes']`` guard is driven with a canned digest."""
    plt.close("all")
    from biosuite.core import cloning as core_cloning

    canned = {"sizes": [], "fragments": [], "cuts": [], "total_bp": len(PLASMID),
              "enzymes_used": ["EcoRI"], "overhangs": {}, "warnings": []}
    monkeypatch.setattr(core_cloning, "simulate_digestion",
                        lambda *a, **k: dict(canned))
    ui.set("cloning_seq", PLASMID)
    ui.app.cloning_enzyme.set("EcoRI")

    ui.app._run_virtual_gel()

    assert ui.messages == [("info", "No Fragments",
                            "EcoRI does not cut this sequence — uncut lane shown.")]
    assert ui.plots, "the uncut lane is still drawn"
    assert ui.app._last_digest_result == canned


# ─── PCR ────────────────────────────────────────────────────────────────────

def test_pcr_writes_the_report_the_core_produces(ui, status_log):
    expected = simulate_pcr(PCR_TEMPLATE, FWD, REV)
    assert expected["size"] == len(PCR_TEMPLATE) - 12, "both primers bind, so the product is nearly all of it"
    ui.set("cloning_seq", PCR_TEMPLATE)
    ui.set("pcr_fwd", FWD)
    ui.set("pcr_rev", REV)
    ui.clear("cloning_result")

    ui.app._run_pcr()

    assert ui.text("cloning_result").strip() == format_pcr_report(expected).strip()
    assert f"{expected['size']} bp" in ui.text("cloning_result")
    assert ui.messages == []
    assert status_log[0] == "Running PCR simulation..."
    assert "PCR simulation complete" in status_log


def test_pcr_without_a_sequence_asks_for_one(ui):
    ui.clear("cloning_seq")
    ui.set("pcr_fwd", FWD)
    ui.set("pcr_rev", REV)

    ui.app._run_pcr()

    assert ui.messages_of("warning") == ["Please enter or load a DNA sequence."]


def test_pcr_insists_on_both_primers(ui):
    ui.set("cloning_seq", PCR_TEMPLATE)
    ui.clear("cloning_result")

    ui.set("pcr_fwd", "")
    ui.set("pcr_rev", REV)
    ui.app._run_pcr()
    ui.set("pcr_fwd", FWD)
    ui.set("pcr_rev", "")
    ui.app._run_pcr()

    assert ui.messages_of("warning") == ["Enter both forward and reverse primers."] * 2
    assert ui.text("cloning_result").strip() == ""


def test_pcr_reports_a_primer_that_is_not_in_the_template(ui):
    ui.set("cloning_seq", PCR_TEMPLATE)
    ui.set("pcr_fwd", "GGGGGGGGGGGGGGGGGGGG")
    ui.set("pcr_rev", REV)
    ui.set("cloning_result", "KEEP THIS REPORT")

    ui.app._run_pcr()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "GGGGGGGGGGGGGGGGGGGG" in errors[0], errors
    assert ui.text("cloning_result").strip() == "KEEP THIS REPORT"


# ─── Primer design ──────────────────────────────────────────────────────────

def test_primer_design_writes_the_report_the_core_produces(ui, status_log):
    expected = design_primers(PRIMER_TARGET)
    assert len(expected["forward"]) == 20 and len(expected["reverse"]) == 20
    ui.set("primer_seq", PRIMER_TARGET)
    ui.clear("cloning_result")

    ui.app._design_primers()

    assert ui.text("cloning_result").strip() == format_primer_report(expected).strip()
    assert f"5'-{expected['forward']}-3'" in ui.text("cloning_result")
    assert ui.messages == []
    assert status_log[0] == "Designing primers..."
    assert "Primers designed" in status_log


def test_primer_design_needs_a_target(ui, status_log):
    ui.clear("primer_seq")
    ui.clear("cloning_result")

    ui.app._design_primers()

    assert ui.messages_of("warning") == ["Enter a target sequence for primer design."]
    assert status_log == []
    assert ui.text("cloning_result").strip() == ""


def test_primer_design_reports_a_template_too_short_to_flank(ui):
    short = "ATGCGTGCCAAA"
    assert len(short) < 20
    with pytest.raises(ValueError, match="shorter than primer length"):
        design_primers(short)
    ui.set("primer_seq", short)
    ui.set("cloning_result", "KEEP THIS REPORT")

    ui.app._design_primers()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and f"{len(short)} bp" in errors[0], errors
    assert ui.text("cloning_result").strip() == "KEEP THIS REPORT"


# ─── Insert verification ────────────────────────────────────────────────────

INSERT = "ATGCGGCCATGCAAACGT"
BACKBONE = "GAATTCGGATCCACGTGCGGCCGC"


def _insert_verdict_text(result):
    lines = ["Insert verification:\n",
             f"Insert found: {result['insert_found']}\n",
             f"Forward orientation: {result['forward_orientation']}\n"]
    if result['insert_start'] is not None:
        lines.append(f"Position: {result['insert_start']}-{result['insert_end']}\n")
    lines.extend(f"Warning: {w}\n" for w in result['warnings'])
    return "".join(lines).rstrip("\n")


def test_insert_verification_needs_a_product(ui):
    ui.clear("cloning_seq")
    ui.set("clone_insert", INSERT)
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.messages_of("warning") == ["Enter the ligated product sequence first."]
    assert ui.text("cloning_result").strip() == ""


def test_insert_verification_needs_an_insert(ui):
    ui.set("cloning_seq", BACKBONE + INSERT + BACKBONE)
    ui.clear("clone_insert")
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.messages_of("warning") == ["Enter the expected insert sequence."]
    assert ui.text("cloning_result").strip() == ""


def test_insert_verification_reports_the_core_verdict(ui):
    product = BACKBONE + INSERT
    expected = verify_insert(product, INSERT, BACKBONE)
    # Positive control: the core found the insert in the forward orientation.
    assert expected["insert_found"] is True
    assert expected["forward_orientation"] is True
    assert expected["warnings"] == []
    ui.set("cloning_seq", product)
    ui.set("clone_insert", INSERT)
    ui.set("clone_backbone", BACKBONE)
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.text("cloning_result").rstrip("\n") == _insert_verdict_text(expected)
    assert f"Position: {len(BACKBONE)}-{len(BACKBONE) + len(INSERT)}" in ui.text("cloning_result")
    assert ui.status() == "Insert verification complete"
    assert ui.messages == []


def test_insert_verification_finds_a_reverse_insert(ui):
    product = BACKBONE + _reverse_complement(INSERT) + BACKBONE
    expected = verify_insert(product, INSERT, BACKBONE)
    assert expected["insert_found"] is True
    assert expected["forward_orientation"] is False
    ui.set("cloning_seq", product)
    ui.set("clone_insert", INSERT)
    ui.set("clone_backbone", BACKBONE)
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.text("cloning_result").rstrip("\n") == _insert_verdict_text(expected)
    assert "Forward orientation: False" in ui.text("cloning_result")


def test_insert_verification_reports_a_missing_insert(ui):
    product = BACKBONE + BACKBONE
    expected = verify_insert(product, INSERT)
    assert expected["insert_found"] is False
    assert "Insert NOT found in the product sequence." in expected["warnings"]
    ui.set("cloning_seq", product)
    ui.set("clone_insert", INSERT)
    ui.clear("clone_backbone")
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.text("cloning_result").rstrip("\n") == _insert_verdict_text(expected)
    assert "Warning: Insert NOT found in the product sequence." in ui.text("cloning_result")


def test_insert_verification_with_a_wrong_backbone_warns(ui):
    product = "TTTTTTTT" + INSERT + "TTTTTTTT"
    expected = verify_insert(product, INSERT, BACKBONE)
    assert expected["warnings"], "the core must flag flanks that are not the backbone"
    ui.set("cloning_seq", product)
    ui.set("clone_insert", INSERT)
    ui.set("clone_backbone", BACKBONE)
    ui.clear("cloning_result")

    ui.app._verify_insert()

    assert ui.text("cloning_result").rstrip("\n") == _insert_verdict_text(expected)
    for warning in expected["warnings"]:
        assert f"Warning: {warning}" in ui.text("cloning_result")


def test_insert_verification_names_invalid_dna(ui):
    with pytest.raises(ValueError) as exc:
        verify_insert("ACGT!", INSERT)
    ui.set("cloning_seq", "ACGT!")
    ui.set("clone_insert", INSERT)
    ui.clear("clone_backbone")
    ui.set("cloning_result", "KEEP THIS REPORT")

    ui.app._verify_insert()

    assert ui.messages_of("error") == [str(exc.value)]
    assert ui.text("cloning_result").strip() == "KEEP THIS REPORT"
