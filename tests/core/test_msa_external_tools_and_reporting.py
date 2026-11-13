"""Contracts pinned here: every boundary of ``biosuite/core/msa.py`` that the
pure-Python aligner does not exercise — the external Clustal Omega / MUSCLE /
MAFFT wrappers, the no-Biopython fallbacks, the reporting layer
(conservation, consensus, statistics, text rendering) and the degenerate
inputs that used to be answered with a silent zero.

Why it matters: an MSA result is consumed by the phylogeny, plotting and GUI
code as if its rows were ordered, its lengths consistent and its conservation
vector real. A wrapper that quietly returned ``None``, a formatter that printed
a header without the alignment, or a conservation vector of a different length
all looked like successful alignments from the outside.
"""
import contextlib
import importlib.util
import os
import subprocess
import sys

import pytest

from biosuite.core import msa as ms

#: A valid aligned FASTA whose rows are NOT in the order the tool was called with.
ALIGNED_FASTA = ">beta\nAC-GTA\n>alpha\nACG-TA\n"

#: Rows of a protein pair whose honest alignment keeps five identical columns.
PROTEIN_PAIR = ["MKVLAG"], ["MKLAG"]


# ── helpers ──────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _executed_without(*packages):
    """Run msa.py again with ``packages`` hidden and yield the resulting module.

    ``HAS_BIO`` is decided while the module body runs, so the only way to reach
    the ``except ImportError`` branch and the parser fallbacks underneath it is
    to execute the file again with Biopython invisible.  A fresh module object is
    used rather than ``importlib.reload`` so the shared ``biosuite.core.msa``
    keeps whatever it had already imported.
    """
    for package in packages:
        pytest.importorskip(package)      # hiding something absent proves nothing

    class HiddenFinder:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in packages:
                raise ImportError(f"{name} is hidden by this test")
            return None

    finder = HiddenFinder()
    parked = {name: module for name, module in sys.modules.items()
              if name.split(".")[0] in packages}
    for name in parked:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    # A dotted alias is what gives the fresh module a __package__, so the
    # module's own `from .log import get_logger` still resolves.
    alias = "biosuite.core._msa_without_" + "_and_".join(packages)
    spec = importlib.util.spec_from_file_location(alias, ms.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(parked)
        sys.modules.pop(alias, None)


class _CompletedProcess:
    """The three attributes of subprocess.CompletedProcess the wrappers read."""

    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeToolRun:
    """Stand-in for subprocess.run that plays the part of an MSA binary.

    ``responses`` is consumed per invocation, so MUSCLE's two-CLI retry can be
    scripted.  When a response carries stdout and the command names an output
    flag, the stdout is written to that path exactly as the real tool would.
    """

    def __init__(self, responses, output_flags=("-o",), error=None, consume_output=False):
        self.responses = list(responses)
        self.output_flags = output_flags
        self.error = error
        self.consume_output = consume_output
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if self.error is not None:
            raise self.error
        returncode, stdout = self.responses[min(len(self.commands) - 1, len(self.responses) - 1)]
        for flag in self.output_flags:
            if flag in cmd and stdout:
                path = cmd[cmd.index(flag) + 1]
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(stdout)
                if self.consume_output:
                    os.unlink(path)
        return _CompletedProcess(returncode, stdout)


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    """Keep every scratch file the tool wrappers create inside pytest's tmp_path.

    The wrappers deliberately reserve the output path by creating it, which is
    why they must not trust ``os.path.exists``; the fake ``secure_temp_path``
    reproduces that reservation so the tests measure the same thing.
    """
    counter = {"n": 0}
    real_named = ms.tempfile.NamedTemporaryFile

    def named(*args, suffix="", **kwargs):
        return real_named(dir=str(tmp_path), suffix=suffix, **kwargs)

    def secure(suffix="", prefix="biosuite_"):
        counter["n"] += 1
        path = tmp_path / f"{prefix}{counter['n']}{suffix}"
        path.write_text("")
        return str(path)

    monkeypatch.setattr(ms.tempfile, "NamedTemporaryFile", named)
    monkeypatch.setattr(ms, "secure_temp_path", secure)
    return tmp_path


def _msa_with_conservation(pairs, alignment_length=None, conservation=None, **kwargs):
    """Build an MSA the way auto_align would, so no test depends on defaults."""
    sequences = list(pairs)
    return ms.MSA(
        method=kwargs.pop("method", "builtin"),
        sequences=sequences,
        num_sequences=kwargs.pop("num_sequences", len(sequences)),
        alignment_length=alignment_length if alignment_length is not None
        else max((len(s) for _, s in sequences), default=0),
        conservation=[] if conservation is None else list(conservation),
        **kwargs)


# ── the no-Biopython side of the module ──────────────────────────────────────

def test_msa_imports_and_reports_no_biopython_when_bio_cannot_be_found():
    with _executed_without("Bio") as fresh:
        assert fresh.HAS_BIO is False, "the ImportError guard did not take effect"


def test_the_fallback_parser_reads_multiline_records_and_keeps_file_order(tmp_path):
    path = tmp_path / "in.fa"
    path.write_text(">alpha first sequence\nACG\nT\n\n>beta\nGGG\n", encoding="utf-8")
    with _executed_without("Bio") as fresh:
        assert fresh.read_fasta_for_msa(str(path)) == [("alpha", "ACGT"), ("beta", "GGG")]


def test_the_fallback_parser_returns_nothing_for_a_headerless_file(tmp_path):
    path = tmp_path / "bare.fa"
    path.write_text("\n   \n", encoding="utf-8")
    with _executed_without("Bio") as fresh:
        assert fresh.read_fasta_for_msa(str(path)) == []


def test_the_fallback_parser_drops_residues_that_precede_the_first_header(tmp_path):
    """Pinned, not endorsed: without Biopython a leading sequence line is lost.

    SeqIO.parse raises ValueError on the same file, so the two parser paths
    disagree about whether a headerless FASTA is an error or a partial result.
    """
    path = tmp_path / "orphan.fa"
    path.write_text("ACGT\n>only\nTT\n", encoding="utf-8")
    with _executed_without("Bio") as fresh:
        assert fresh.read_fasta_for_msa(str(path)) == [("only", "TT")]


def test_loading_an_alignment_from_disk_is_refused_without_biopython(tmp_path):
    path = tmp_path / "aln.fa"
    path.write_text(ALIGNED_FASTA, encoding="utf-8")
    with _executed_without("Bio") as fresh:
        assert fresh._load_bio_alignment(str(path), "muscle") is None


def test_the_builtin_engine_still_aligns_without_biopython():
    with _executed_without("Bio") as fresh:
        result = fresh.auto_align(["ACGT", "ACGA"])
        assert result.sequences == [("seq1", "ACGT"), ("seq2", "ACGA")]
        assert result.conservation == [1.0, 1.0, 1.0, 0.5]
        assert result.engine == "builtin"


def test_the_session_module_is_untouched_by_hiding_biopython():
    """The reason a fresh module object is used instead of reloading the real one."""
    pytest.importorskip("Bio")
    assert ms.HAS_BIO is True
    assert ms.auto_align(["ACGT", "ACGA"]).sequences == [("seq1", "ACGT"), ("seq2", "ACGA")]


def test_biopython_and_the_fallback_parser_agree_on_a_clean_fasta(tmp_path):
    """Both halves of read_fasta_for_msa must answer the same way."""
    path = tmp_path / "in.fa"
    text = ">alpha first sequence\nACG\nT\n\n>beta\nGGG\n"
    path.write_text(text, encoding="utf-8")
    with _executed_without("Bio") as fresh:
        fallback = fresh.read_fasta_for_msa(str(path))
    assert ms.HAS_BIO is True
    assert ms.read_fasta_for_msa(str(path)) == fallback == [("alpha", "ACGT"), ("beta", "GGG")]


# ── sequence classification and distance ─────────────────────────────────────

def test_an_empty_sequence_is_classified_as_nucleotide():
    assert ms._is_nucleotide("") is True
    assert ms._is_nucleotide("ATCGATCGATCG-") is True, "one gap in twelve bases is still DNA"
    assert ms._is_nucleotide("ATCG") is True
    assert ms._is_nucleotide("MKVAL") is False


@pytest.mark.parametrize("first, second, k, expected", [
    ("ACGT", "", 3, 1.0),
    ("", "ACGT", 3, 1.0),
    ("", "", 3, 1.0),
    ("AC", "ACGT", 5, 1.0),        # k longer than the words yields no k-mers
    ("ACGT", "ACGT", 3, 0.0),
    ("ACGT", "TTTT", 3, pytest.approx(1.0)),
])
def test_kmer_distance_is_one_when_no_kmer_can_be_compared(first, second, k, expected):
    assert ms._kmer_distance(first, second, k) == expected


# ── pairwise dynamic programming ─────────────────────────────────────────────

def test_a_gap_in_either_input_row_is_scored_as_a_gap_not_a_mismatch():
    aligned_a, aligned_b = ms._align_two_profiles("ACGT", "AC-GT")
    assert (aligned_a, aligned_b) == ("AC-GT", "AC-GT")
    reversed_pair = ms._align_two_profiles("AC-GT", "ACGT")
    assert reversed_pair == ("AC-GT", "AC-GT")
    assert len(aligned_a) == len(aligned_b)


def test_extra_residues_of_the_longer_row_become_gaps_in_the_shorter_row():
    assert ms._align_two_profiles("A", "AC") == ("A-", "AC")
    assert ms._align_two_profiles("AC", "ACGT") == ("AC--", "ACGT")
    assert ms._align_two_profiles("ACGT", "AC") == ("ACGT", "AC--")


def test_pairwise_alignment_against_an_empty_row_is_all_gaps():
    assert ms._align_two_profiles("", "ACGT") == ("----", "ACGT")
    assert ms._align_two_profiles("ACGT", "") == ("ACGT", "----")


# ── profile construction and profile-profile merging ─────────────────────────

def test_merging_an_empty_side_returns_the_other_side_unchanged():
    assert ms._merge_alignments([], ["ACGT"]) == ["ACGT"]
    assert ms._merge_alignments(["ACGT"], []) == ["ACGT"]


def test_a_profile_of_no_sequences_is_an_empty_mapping():
    assert ms._make_profile([]) == {}


def test_profile_frequencies_cover_every_letter_present_in_the_block():
    profiles = ms._make_profile(["AC-T"])
    assert len(profiles) == 4
    assert profiles[0] == {"A": 1.0, "C": 0.0, "T": 0.0, "-": 0.0}
    assert profiles[2]["-"] == 1.0
    assert set(profiles[3]) == {"A", "C", "T", "-"}


def test_identical_profile_columns_reward_a_match_and_different_ones_score_zero():
    one = ms._make_profile(["MKL"])
    assert ms._profile_score_col(one, 0, one, 0) == 10
    assert ms._profile_score_col(one, 0, one, 1) == 0      # M against K
    gaps = ms._make_profile(["--"])
    assert ms._profile_score_col(gaps, 0, gaps, 0) == -20
    assert ms._profile_score_col(one, 0, gaps, 0) == 0     # residue against gap


def test_a_protein_pair_is_merged_so_the_most_columns_agree():
    """The regression: the profile alphabet was nucleotide-only.

    With set('ACGTUNacgtun') every amino-acid column scored 0, so the profile DP
    broke every tie arbitrarily and returned '-MKLAG' (three matching columns)
    for MKVLAG vs MKLAG instead of the five-column answer.
    """
    aligned_a, aligned_b = PROTEIN_PAIR
    merged = ms._merge_alignments(aligned_a, aligned_b)
    assert merged == ["MKVLAG", "MK-LAG"]
    agreements = sum(1 for x, y in zip(merged[0], merged[1]) if x == y and x != "-")
    assert agreements == 5


def test_progressive_alignment_of_three_proteins_places_the_deletion_in_context():
    result = ms.auto_align(["MKVLAGTAA", "MKVLAGTAA", "MKLAGTAA"])
    assert [name for name, _ in result.sequences] == ["seq1", "seq2", "seq3"]
    assert result.sequences_only == ["MKVLAGTAA", "MKVLAGTAA", "MK-LAGTAA"]
    assert result.alignment_length == 9


def test_progressive_alignment_of_two_or_fewer_sequences_returns_them_as_given():
    assert ms._progressive_msa([]) == []
    assert ms._progressive_msa([("a", "ACGT")]) == [("a", "ACGT")]
    assert ms._progressive_msa([("a", "ACGT"), ("b", "ACGA")]) == [("a", "ACGT"), ("b", "ACGA")]


def test_a_merge_that_leads_the_other_side_with_gaps_walks_the_insertion_branch():
    """'-A-' against 'GAC': column 0 and column 2 of b have nothing to pair with."""
    assert ms._merge_alignments(["A"], ["GAC"]) == ["-A-", "GAC"]
    assert ms._merge_alignments(["AC"], ["GAC"]) == ["-AC", "GAC"]


def test_two_fully_gapped_blocks_are_never_aligned_column_by_column():
    """Pinned, not endorsed: a gap column facing a gap column costs -2.

    _profile_score_col charges pa['-'] * pb['-'] * -2, so the DP would rather
    split the two blocks apart than line their empty columns up, and the merged
    alignment grows from two columns to four.  Nothing in the module documents
    that as intentional, so the number is recorded here instead of changed.
    """
    assert ms._merge_alignments(["--"], ["--"]) == ["----", "----"]


# ── UPGMA guide tree ─────────────────────────────────────────────────────────

def test_the_guide_order_lists_every_sequence_exactly_once():
    distances = ms._pairwise_distance(["ACGT", "ACGA", "TCGT", "TTTT"])
    order = ms._build_guide_order(distances, ["a", "b", "c", "d"])
    assert sorted(order) == [0, 1, 2, 3]
    assert len(order) == 4


def test_the_guide_order_walks_the_tree_from_the_shallowest_child_upward():
    """Ties are broken by leaf index, so this traversal order is reproducible.

    'ACGT'/'ACGA' join first (0.667), that cluster then joins 'TCGT', and the
    outlier 'TTTT' is reached last even though it is leaf 3 in the input.
    """
    distances = ms._pairwise_distance(["ACGT", "ACGA", "TCGT", "TTTT"])
    assert ms._build_guide_order(distances, ["a", "b", "c", "d"]) == [3, 2, 0, 1]


def test_a_single_taxon_guide_order_is_that_taxon():
    assert ms._build_guide_order([[0.0]], ["only"]) == [0]
    assert ms._build_guide_order([[0.0, 0.2], [0.2, 0.0]], ["x", "y"]) == [0, 1]


# ── input order restoration ──────────────────────────────────────────────────

def test_restoring_order_of_nothing_needs_no_input_names():
    assert ms._restore_input_order([], [("a", "ACGT")]) == []


def test_a_row_whose_name_is_missing_from_the_input_keeps_its_relative_place():
    assert ms._restore_input_order([("z", "AC")], [("a", "AC")]) == [("z", "AC")]
    ordered = ms._restore_input_order(
        [("b", "11"), ("z", "22"), ("a", "33")], [("a", ""), ("b", "")])
    assert ordered == [("a", "33"), ("b", "11"), ("z", "22")]


def test_repeated_names_are_consumed_one_row_at_a_time():
    assert ms._restore_input_order([("a", "1"), ("a", "2")], [("a", ""), ("a", "")]) \
        == [("a", "1"), ("a", "2")]


# ── Clustal Omega wrapper ────────────────────────────────────────────────────

def test_clustal_omega_output_is_parsed_with_conservation_and_tool_row_order(scratch, monkeypatch):
    fake = _FakeToolRun([(0, ALIGNED_FASTA)])
    monkeypatch.setattr(ms.subprocess, "run", fake)
    result = ms._run_clustal_omega([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert isinstance(result, ms.MSA)
    assert (result.method, result.engine) == ("clustal_omega", "clustal_omega")
    assert result.sequences == [("beta", "AC-GTA"), ("alpha", "ACG-TA")]
    assert result.alignment_length == 6
    assert result.num_sequences == 2
    assert result.conservation == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    command = fake.commands[0]
    assert command[:2] == ["clustalo", "-i"] and command[command.index("-t") + 1] == "DNA"


def test_clustal_omega_is_told_protein_when_the_first_sequence_is_not_nucleotide(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ">a\nMKV-\n>b\nMK-L\n")]))
    result = ms._run_clustal_omega([("a", "MKVAL"), ("b", "MKLA")])
    assert result.sequences_only == ["MKV-", "MK-L"]


@pytest.mark.parametrize("returncode, stdout", [(3, ""), (0, "")],
                         ids=["tool reports failure", "tool exits 0 without output"])
def test_clustal_omega_returns_none_when_the_tool_produces_no_alignment(scratch, monkeypatch,
                                                                       returncode, stdout):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(returncode, stdout)]))
    assert ms._run_clustal_omega([("a", "ACGT"), ("b", "ACGA")]) is None


def test_clustal_omega_reports_none_when_the_executable_cannot_be_spawned(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run",
                        _FakeToolRun([(0, ALIGNED_FASTA)], error=FileNotFoundError("clustalo")))
    assert ms._run_clustal_omega([("a", "ACGT"), ("b", "ACGA")]) is None


def test_clustal_omega_survives_a_cleanup_that_cannot_remove_the_files(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ALIGNED_FASTA)]))

    def locked(path):
        raise OSError("file locked by another process")

    monkeypatch.setattr(ms.os, "unlink", locked)
    result = ms._run_clustal_omega([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.sequences == [("beta", "AC-GTA"), ("alpha", "ACG-TA")]


def test_clustal_omega_returns_none_when_the_tool_removes_its_own_output(scratch, monkeypatch):
    """A binary that cleans up after itself leaves no file for the cleanup loop.

    ``wrote_output`` is what rejects the run, and the ``os.path.exists`` test in
    the finally block is what stops the cleanup from raising on a gone file.
    """
    monkeypatch.setattr(ms.subprocess, "run",
                        _FakeToolRun([(0, ALIGNED_FASTA)], consume_output=True))
    assert ms._run_clustal_omega([("a", "ACGT"), ("b", "ACGA")]) is None


# ── MUSCLE wrapper ───────────────────────────────────────────────────────────

def test_muscle_new_style_cli_is_used_when_it_succeeds(scratch, monkeypatch):
    fake = _FakeToolRun([(0, ALIGNED_FASTA)], output_flags=("-output", "-out"))
    monkeypatch.setattr(ms.subprocess, "run", fake)
    result = ms._run_muscle([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.method == "muscle"
    assert result.sequences_only == ["AC-GTA", "ACG-TA"]
    assert len(fake.commands) == 1 and fake.commands[0][1] == "-align"


def test_muscle_falls_back_to_the_legacy_cli_when_the_new_flags_are_rejected(scratch, monkeypatch):
    fake = _FakeToolRun([(1, ""), (0, ALIGNED_FASTA)], output_flags=("-output", "-out"))
    monkeypatch.setattr(ms.subprocess, "run", fake)
    result = ms._run_muscle([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.alignment_length == 6
    assert len(fake.commands) == 2
    new_style, legacy = fake.commands
    assert new_style[1::2][:2] == ["-align", "-output"]
    assert legacy[1::2][:2] == ["-in", "-out"]
    assert new_style[2] == legacy[2], "the retry must feed the same FASTA back in"
    assert new_style[4] == legacy[4], "and write to the same reserved output path"


def test_muscle_returns_none_when_both_command_styles_fail(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(1, ""), (2, "")],
                                                           output_flags=("-output", "-out")))
    assert ms._run_muscle([("a", "ACGT"), ("b", "ACGA")]) is None


def test_muscle_returns_none_when_the_legacy_run_writes_an_empty_file(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(1, ""), (0, "")],
                                                           output_flags=("-output", "-out")))
    assert ms._run_muscle([("a", "ACGT"), ("b", "ACGA")]) is None


def test_muscle_reports_none_when_the_timeout_fires(scratch, monkeypatch):
    timeout = subprocess.TimeoutExpired(cmd="muscle", timeout=600)
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ALIGNED_FASTA)], error=timeout))
    assert ms._run_muscle([("a", "ACGT"), ("b", "ACGA")]) is None


def test_muscle_survives_a_cleanup_that_cannot_remove_the_files(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ALIGNED_FASTA)],
                                                           output_flags=("-output", "-out")))
    monkeypatch.setattr(ms.os, "unlink", lambda path: (_ for _ in ()).throw(OSError("locked")))
    result = ms._run_muscle([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.engine == "muscle"


def test_muscle_skips_cleanup_for_an_output_file_the_tool_never_created(scratch, monkeypatch):
    """secure_temp_path reserves a path; a failing muscle leaves it empty.

    The cleanup loop then has to tolerate the reserved path being gone, which is
    the one case where ``os.path.exists`` in the finally block is False.
    """
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(1, ""), (1, "")],
                                                           output_flags=("-output", "-out")))
    never = str(scratch / "absent" / "muscle.fasta")
    monkeypatch.setattr(ms, "secure_temp_path", lambda suffix="", prefix="biosuite_": never)
    assert ms._run_muscle([("a", "ACGT"), ("b", "ACGA")]) is None


# ── MAFFT wrapper ────────────────────────────────────────────────────────────

def test_mafft_alignment_is_taken_from_standard_output(scratch, monkeypatch):
    fake = _FakeToolRun([(0, ALIGNED_FASTA)])
    monkeypatch.setattr(ms.subprocess, "run", fake)
    result = ms._run_mafft([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.method == "mafft"
    assert result.sequences == [("beta", "AC-GTA"), ("alpha", "ACG-TA")]
    assert result.conservation == [1.0] * 6
    assert fake.commands[0][:2] == ["mafft", "--auto"]


@pytest.mark.parametrize("returncode, stdout", [(0, "   \n"), (1, ALIGNED_FASTA)],
                         ids=["exit 0 with blank stdout", "nonzero exit with an alignment"])
def test_mafft_returns_none_unless_a_usable_alignment_reaches_stdout(scratch, monkeypatch,
                                                                    returncode, stdout):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(returncode, stdout)]))
    assert ms._run_mafft([("a", "ACGT"), ("b", "ACGA")]) is None


def test_mafft_reports_none_when_the_executable_is_missing(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ALIGNED_FASTA)],
                                                           error=OSError("no mafft")))
    assert ms._run_mafft([("a", "ACGT"), ("b", "ACGA")]) is None


def test_mafft_survives_a_cleanup_that_cannot_remove_the_files(scratch, monkeypatch):
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(0, ALIGNED_FASTA)]))
    monkeypatch.setattr(ms.os, "unlink", lambda path: (_ for _ in ()).throw(OSError("locked")))
    result = ms._run_mafft([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.alignment_length == 6


def test_mafft_skips_cleanup_for_an_output_file_it_never_wrote_to(scratch, monkeypatch):
    """mafft answers on stdout, so the reserved path stays empty on failure."""
    monkeypatch.setattr(ms.subprocess, "run", _FakeToolRun([(1, ALIGNED_FASTA)]))
    never = str(scratch / "absent" / "mafft.fasta")
    monkeypatch.setattr(ms, "secure_temp_path", lambda suffix="", prefix="biosuite_": never)
    assert ms._run_mafft([("a", "ACGT"), ("b", "ACGA")]) is None


# ── degenerate alignments ────────────────────────────────────────────────────

def test_an_empty_alignment_request_keeps_the_no_op_contract():
    result = ms.auto_align([])
    assert result.method == "none" and result.engine == "builtin"
    assert result.sequences == [] and result.names == []
    assert (result.num_sequences, result.alignment_length) == (0, 0)
    assert result.conservation == []
    assert result.message == "Need at least 2 sequences."


def test_a_single_sequence_is_returned_unaligned_rather_than_dropped():
    result = ms.auto_align(["ACGT"])
    assert result.method == "none"
    assert result.sequences == [("seq1", "ACGT")]
    assert result.num_sequences == 1 and result.alignment_length == 4
    assert result.conservation == [1.0, 1.0, 1.0, 1.0]
    assert result.message == "Need at least 2 sequences."
    assert ms.alignment_statistics(result)["gap_percentage"] == 0.0


# ── parsing a tool's FASTA back into an MSA ──────────────────────────────────

def test_an_unparseable_alignment_is_reported_as_none(tmp_path):
    junk = tmp_path / "junk.fa"
    junk.write_text("this is not fasta\nat all\n", encoding="utf-8")
    assert ms._load_bio_alignment(str(junk), "clustal_omega") is None


def test_a_truncated_alignment_of_unequal_rows_is_reported_as_none(tmp_path):
    broken = tmp_path / "broken.fa"
    broken.write_text(">a\nACGT\n>b\nACGTA\n", encoding="utf-8")
    assert ms._load_bio_alignment(str(broken), "muscle") is None


def test_an_alignment_file_that_was_never_written_is_reported_as_none(tmp_path):
    assert ms._load_bio_alignment(str(tmp_path / "gone.fa"), "mafft") is None


# ── engine choice in auto_align ──────────────────────────────────────────────

def _available_tool(monkeypatch, **available):
    tools = {"clustal_omega": False, "muscle": False, "mafft": False}
    tools.update(available)
    monkeypatch.setattr(ms, "check_tools", lambda: tools)
    return tools


def test_clustal_omega_result_is_returned_with_its_rows_reordered_to_the_input(monkeypatch):
    _available_tool(monkeypatch, clustal_omega=True)
    external = _msa_with_conservation([("beta", "AC-GTA"), ("alpha", "ACG-TA")],
                                      alignment_length=6, method="clustal_omega")
    monkeypatch.setattr(ms, "_run_clustal_omega", lambda seqs: external)
    result = ms.auto_align([("alpha", "ACGT"), ("beta", "ACGTA")])
    assert result.sequences == [("alpha", "ACG-TA"), ("beta", "AC-GTA")]
    assert result.names == ["alpha", "beta"]
    assert result.sequences_only == ["ACG-TA", "AC-GTA"]
    assert result.message == "Using Clustal Omega (external)"
    assert result.method == "clustal_omega"


def test_a_failed_clustal_omega_run_falls_through_to_muscle(monkeypatch):
    _available_tool(monkeypatch, clustal_omega=True, muscle=True)
    monkeypatch.setattr(ms, "_run_clustal_omega", lambda seqs: None)
    monkeypatch.setattr(ms, "_run_muscle",
                        lambda seqs: _msa_with_conservation([("a", "ACGT"), ("b", "ACGA")],
                                                            method="muscle"))
    result = ms.auto_align(["ACGT", "ACGA"])
    assert result.method == "muscle"
    assert result.message == "Using MUSCLE (external)"


def test_a_failed_muscle_run_falls_through_to_mafft(monkeypatch):
    _available_tool(monkeypatch, muscle=True, mafft=True)
    monkeypatch.setattr(ms, "_run_muscle", lambda seqs: None)
    monkeypatch.setattr(ms, "_run_mafft",
                        lambda seqs: _msa_with_conservation([("b", "AC-GTA"), ("a", "ACG-T")],
                                                            method="mafft"))
    result = ms.auto_align([("a", "ACGT"), ("b", "ACGA")])
    assert result.method == "mafft"
    assert result.sequences == [("a", "ACG-T"), ("b", "AC-GTA")]
    assert result.message == "Using MAFFT (external)"


def test_a_failed_mafft_run_falls_back_to_the_builtin_engine(monkeypatch):
    _available_tool(monkeypatch, clustal_omega=True, muscle=True, mafft=True)
    monkeypatch.setattr(ms, "_run_clustal_omega", lambda seqs: None)
    monkeypatch.setattr(ms, "_run_muscle", lambda seqs: None)
    monkeypatch.setattr(ms, "_run_mafft", lambda seqs: None)
    result = ms.auto_align(["ACGT", "ACGA"])
    assert result.method == "builtin_progressive"
    assert result.engine == "builtin"
    assert result.message == "Using built-in progressive alignment engine"
    assert result.conservation == [1.0, 1.0, 1.0, 0.5]


def test_an_external_result_whose_conservation_length_disagrees_is_rejected(monkeypatch):
    """The guard is defensive: no engine can produce the mismatch today.

    compute_conservation returns exactly one score per column of
    alignment_length, so this only fires if the two ever drift apart again —
    which is the failure the module already had once, hence the explicit check.
    """
    _available_tool(monkeypatch)
    monkeypatch.setattr(ms, "compute_conservation", lambda alignment: [1.0, 1.0])
    with pytest.raises(RuntimeError, match=r"conservation vector length 2 does not "
                                           r"match alignment length 4"):
        ms.auto_align(["ACGT", "ACGA"])


# ── the explicit engine wrappers ─────────────────────────────────────────────

WRAPPERS = [(ms.run_clustal_omega, "clustal_omega"), (ms.run_muscle, "muscle"),
            (ms.run_mafft, "mafft")]


@pytest.mark.parametrize("wrapper, engine", WRAPPERS, ids=["clustal", "muscle", "mafft"])
def test_a_named_wrapper_refuses_an_engine_that_is_not_installed(monkeypatch, wrapper, engine):
    """The docstrings have always promised a refusal.  They used to fall through to
    whatever auto_align picked, so on this install (no external binaries) all three
    returned the built-in progressive alignment and reported nothing was wrong."""
    _available_tool(monkeypatch)
    with pytest.raises(RuntimeError,
                       match=rf"{engine} is not installed; it was requested explicitly"):
        wrapper(["ACGT", "ACGA"])


@pytest.mark.parametrize("wrapper, engine", WRAPPERS, ids=["clustal", "muscle", "mafft"])
def test_a_named_wrapper_uses_the_engine_it_names(monkeypatch, wrapper, engine):
    """Requesting MUSCLE must not silently return Clustal Omega's answer."""
    _available_tool(monkeypatch, clustal_omega=True, muscle=True, mafft=True)
    for name in ("clustal_omega", "muscle", "mafft"):
        monkeypatch.setattr(
            ms, f"_run_{name}",
            lambda seqs, _n=name: _msa_with_conservation(
                [("seq1", "ACGT"), ("seq2", "ACGA")], alignment_length=4, method=_n))
    result = wrapper(["ACGT", "ACGA"])
    assert result.method == engine
    assert result.message in ("Using Clustal Omega (external)", "Using MUSCLE (external)",
                              "Using MAFFT (external)"), result.message
    assert result.sequences == [("seq1", "ACGT"), ("seq2", "ACGA")]


@pytest.mark.parametrize("wrapper, engine", WRAPPERS, ids=["clustal", "muscle", "mafft"])
def test_a_named_wrapper_rejects_options_it_cannot_forward(wrapper, engine, monkeypatch):
    """threads= and align= were taken by **kwargs and never read, so a caller that
    set them believed the engine had been configured."""
    _available_tool(monkeypatch, **{engine: True})
    with pytest.raises(TypeError, match="unexpected keyword argument 'threads'"):
        wrapper(["ACGT", "ACGA"], threads=8)


def test_auto_align_refuses_an_unknown_method_name():
    with pytest.raises(ValueError,
                       match=r"Unknown method: progressive\. Use 'auto', 'builtin' or one of "
                             r"clustal_omega, muscle, mafft\."):
        ms.auto_align(["ACGT", "ACGA"], method="progressive")


def test_method_builtin_skips_the_external_engines(monkeypatch):
    """'builtin' was accepted before only by falling through; now it is named, and
    an installed external engine must not be used."""
    _available_tool(monkeypatch, clustal_omega=True)
    monkeypatch.setattr(ms, "_run_clustal_omega",
                        lambda seqs: pytest.fail("the external engine must not run"))
    result = ms.auto_align(["ACGT", "ACGA"], method="builtin")
    assert result.method == "builtin_progressive"
    assert result.message == "Using built-in progressive alignment engine"


def test_a_requested_engine_that_installs_but_returns_nothing_raises(monkeypatch):
    _available_tool(monkeypatch, muscle=True)
    monkeypatch.setattr(ms, "_run_muscle", lambda seqs: None)
    with pytest.raises(RuntimeError,
                       match="muscle is installed but produced no alignment"):
        ms.auto_align(["ACGT", "ACGA"], method="muscle")


def test_check_tools_reports_the_three_binaries_this_install_can_run():
    detected = ms.check_tools()
    assert sorted(detected) == ["clustal_omega", "mafft", "muscle"]
    assert all(isinstance(value, bool) for value in detected.values())


# ── input normalisation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", ["ACGTACGT", b"ACGTACGT"], ids=["str", "bytes"])
def test_a_bare_fasta_payload_is_rejected_before_it_can_be_aligned_as_letters(payload):
    with pytest.raises(TypeError, match="not a single string"):
        ms._normalize_sequences(payload)
    with pytest.raises(TypeError, match="To align a FASTA file"):
        ms.auto_align(payload)


def test_unnamed_sequences_are_numbered_from_one_and_named_pairs_are_kept():
    assert ms._normalize_sequences(["ACGT", ("x", "ACGA")]) == [("seq1", "ACGT"), ("x", "ACGA")]


def test_fasta_written_for_a_tool_reuses_the_normalised_names(tmp_path):
    path = tmp_path / "out.fa"
    ms._write_fasta([("x", "ACGT"), "ACGA"], str(path))
    assert path.read_text(encoding="utf-8") == ">x\nACGT\n>seq2\nACGA\n"


# ── conservation, consensus, statistics, rendering ───────────────────────────

def test_conservation_of_a_missing_or_empty_alignment_is_an_empty_vector():
    assert ms.compute_conservation(None) == []
    assert ms.compute_conservation(_msa_with_conservation([])) == []


def test_a_column_of_only_gains_no_conservation_credit():
    alignment = _msa_with_conservation([("a", "--"), ("b", "--")])
    assert ms.compute_conservation(alignment) == [0.0, 0.0]


def test_conservation_is_the_majority_share_of_the_observed_residues():
    alignment = _msa_with_conservation([("a", "ACGT"), ("b", "ACGA"), ("c", "ACGC")])
    assert ms.compute_conservation(alignment) == [1.0, 1.0, 1.0, pytest.approx(1 / 3)]


def test_conservation_ignores_gaps_so_a_partially_observed_column_reads_as_conserved():
    """Pinned, not endorsed: three of four rows are gaps and the column still
    scores 1.0, and the alignment as a whole reports mean 1.0 at 50% gaps."""
    alignment = _msa_with_conservation([("a", "A-GT"), ("b", "-G-T"), ("c", "--G-")])
    assert ms.compute_conservation(alignment) == [1.0, 1.0, 1.0, 1.0]
    assert ms.alignment_statistics(alignment)["gap_percentage"] == 50.0


def test_consensus_of_an_alignment_without_sequences_is_empty():
    assert ms.consensus_sequence(None) == ""
    assert ms.consensus_sequence(_msa_with_conservation([("a", "AC")], num_sequences=0)) == ""


def test_a_column_of_only_gaps_is_a_gap_in_the_consensus():
    alignment = _msa_with_conservation([("a", "-C"), ("b", "--")])
    assert ms.consensus_sequence(alignment, threshold=0.5) == "-C"


@pytest.mark.parametrize("threshold, expected", [
    (0.5, "ACGt"),      # 2 of 3 rows carry T: a majority, so lower-case
    (1.0, "ACGn"),      # nothing reaches certainty: the ambiguity letter
], ids=["majority rule", "certainty required"])
def test_a_residue_below_the_threshold_becomes_n_and_a_partial_majority_is_lowercased(
        threshold, expected):
    alignment = _msa_with_conservation([("a", "ACGT"), ("b", "ACGT"), ("c", "ACGA")])
    assert ms.consensus_sequence(alignment, threshold=threshold) == expected


def test_a_column_with_three_different_residues_below_half_is_n():
    alignment = _msa_with_conservation([("a", "ACGT"), ("b", "ACGA"), ("c", "ACGC")])
    assert ms.consensus_sequence(alignment, threshold=0.5) == "ACGn"


def test_statistics_of_no_alignment_are_an_empty_mapping():
    assert ms.alignment_statistics(None) == {}


def test_statistics_are_computed_on_demand_when_the_engine_left_conservation_empty():
    alignment = _msa_with_conservation([("a", "ACGT"), ("b", "ACGA"), ("c", "ACGC")],
                                       conservation=[])
    stats = ms.alignment_statistics(alignment)
    assert stats == {"num_sequences": 3, "alignment_length": 4, "gap_percentage": 0.0,
                     "mean_conservation": 0.8333, "highly_conserved": 3, "variable": 1}


def test_statistics_of_an_empty_alignment_report_zero_rather_than_dividing_by_zero():
    stats = ms.alignment_statistics(_msa_with_conservation([], num_sequences=0,
                                                           alignment_length=0))
    assert stats["gap_percentage"] == 0
    assert stats["mean_conservation"] == 0
    assert stats["highly_conserved"] == 0 and stats["variable"] == 0


def test_an_alignment_without_rows_says_so_instead_of_printing_a_header():
    assert ms.format_alignment(None) == "No alignment to display."
    assert ms.format_alignment(_msa_with_conservation([])) == "No alignment to display."


def test_a_block_renders_every_row_and_a_conservation_bar_of_the_right_glyphs():
    alignment = _msa_with_conservation([("alpha", "ACGTACGTA"), ("beta", "ACGTA-CGT")],
                                       conservation=[1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 1.0, 1.0, 1.0],
                                       message="Using MUSCLE (external)")
    assert alignment.alignment_length == 9
    assert ms.format_alignment(alignment, max_width=80) == (
        "Engine: Using MUSCLE (external)\nMethod: builtin\nSequences: 2\nLength: 9\n"
        "\nalpha  ACGTACGTA\n beta  ACGTA-CGT\n       |||:. |||\n")


def test_the_conservation_row_is_omitted_when_asked_or_when_nothing_was_scored():
    alignment = _msa_with_conservation([("a", "AC")], conservation=[1.0, 1.0])
    hidden = ms.format_alignment(alignment, show_conservation=False)
    assert hidden.endswith("a  AC\n")
    assert "|" not in hidden
    without_scores = ms.format_alignment(_msa_with_conservation([("a", "AC")]))
    assert without_scores.endswith("a  AC\n")
    assert "|" not in without_scores


def test_a_name_as_long_as_the_block_width_still_prints_the_whole_alignment():
    """The regression: seq_width went to zero or negative.

    ``range(0, length, 0)`` raised ValueError, and a negative step produced an
    empty loop that printed the header only - both for a 56-character sequence
    identifier, which real FASTA headers routinely are.
    """
    long_name = "NCBI_ANNOTATION_" + "X" * 40
    alignment = _msa_with_conservation([(long_name, "ACGTACGTAC")],
                                       conservation=[1.0] * 10, message="engine")
    rendered = ms.format_alignment(alignment, max_width=60)
    residues = "".join(line[len(long_name) + 2:] for line in rendered.splitlines()
                       if line.startswith(long_name))
    assert residues == "ACGTACGTAC"
    assert rendered.count("|") == 10
    seventy_six = _msa_with_conservation([("a" * 76, "ACGT")])
    wide = ms.format_alignment(seventy_six, max_width=80)
    assert wide.count("ACGT") == 0 and "".join(
        line[-1] for line in wide.splitlines() if line.startswith("a" * 76)) == "ACGT"
