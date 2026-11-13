"""The notebook module's non-magic surface: import flags and the ``quick_*`` helpers.

``quick_gc``, ``quick_translate``, ``quick_align`` and ``quick_blast`` exist so a
notebook can do a one-line analysis without wiring up magic commands.  They are thin
adapters over ``biosuite.core``, so each test here asserts the value the caller sees,
not that a function exists.  The magic commands live in ``test_notebook_magics.py``
and the ipywidgets classes in ``test_notebook_widgets.py``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repository root importable when the tests run from a checkout without
# the package installed.  Resolved relative to this file so it works on any machine.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import biosuite.notebook as nbmod  # noqa: E402

from biosuite.notebook import (  # noqa: E402
    quick_align,
    quick_blast,
    quick_gc,
    quick_translate,
)


# ── Module import ────────────────────────────────────────────────────────────

def test_the_module_imports_whether_or_not_ipython_is_present():
    assert isinstance(nbmod.HAS_IPYTHON, bool)
    assert isinstance(nbmod.HAS_WIDGETS, bool)
    assert nbmod.quick_gc("ATCG") == 50.0


# ── quick_gc ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sequence,expected", [
    ("ATCGATCG", 50.0),
    ("GCGCGCGC", 100.0),
    ("ATATAT", 0.0),
    ("", 0.0),
    ("atcgatcg", 50.0),
])
def test_quick_gc_reports_the_percentage(sequence, expected):
    assert quick_gc(sequence) == pytest.approx(expected)


def test_quick_gc_handles_a_generated_sequence():
    seq = "".join(np.random.default_rng(7).choice(["A", "T", "G", "C"], size=1000))

    assert 0.0 <= quick_gc(seq) <= 100.0


# ── quick_translate ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("codons,protein", [
    ("ATG", "M"),
    ("TAA", "*"),
    ("ATGAAATTTTAA", "MKF*"),
    ("ATGNNNTAA", "MX*"),
])
def test_quick_translate_reads_the_frame_it_is_given(codons, protein):
    assert quick_translate(codons) == protein


def test_quick_translate_refuses_a_protein_sequence():
    """A protein handed in by mistake used to come back as a plausible string of
    X's; an amino-acid-only letter is not DNA, so the call must fail instead."""
    with pytest.raises(ValueError, match="IUPAC"):
        quick_translate("MKFLVQPWX")


def test_quick_translate_of_a_generated_sequence():
    np.random.seed(42)
    seq = "".join(np.random.choice(["A", "T", "G", "C"], size=300))

    assert len(quick_translate(seq)) == 100


# ── quick_align ──────────────────────────────────────────────────────────────

def test_quick_align_scores_identical_sequences_globally_and_locally():
    a1, a2, global_score = quick_align("ATCG", "ATCG", method="nw")
    l1, l2, local_score = quick_align("ATCG", "ATCG", method="sw")

    assert (a1, a2) == ("ATCG", "ATCG")
    assert global_score > 0 and local_score > 0


def test_quick_align_of_different_lengths_pads_both_rows():
    a1, a2, _score = quick_align("AGTACGCA", "TATGC", method="nw")

    assert len(a1) == len(a2) == 8
    assert a2.count("-") == 3


def test_quick_align_finds_the_local_match_between_the_x_padding():
    _a1, _a2, score = quick_align("XXXATCGYYY", "ATCG", method="sw")

    assert score > 0


def test_quick_align_defaults_to_needleman_wunsch():
    defaulted = quick_align("ATCGATCG", "ATCAATCG")
    forced = quick_align("ATCGATCG", "ATCAATCG", method="nw")

    assert defaulted == forced


def test_quick_align_refuses_a_method_it_does_not_have():
    """Any spelling that is not 'nw' used to mean Smith-Waterman; a typo must be
    named, not silently re-routed."""
    with pytest.raises(ValueError, match="use 'nw' or 'sw'"):
        quick_align("ACGT", "ACGT", method="nw-ish")


def test_quick_align_returns_a_numeric_score_for_generated_sequences():
    np.random.seed(123)
    s1 = "".join(np.random.choice(list("ACGT"), size=50))
    s2 = "".join(np.random.choice(list("ACGT"), size=30))

    _a1, _a2, score = quick_align(s1, s2, method="nw")

    assert type(score) is int
    assert score == -29  # the seeded pair's measured global score


# ── quick_blast ──────────────────────────────────────────────────────────────

@pytest.fixture
def blast_files(tmp_path):
    """A 60 bp query and a database holding it plus one unrelated subject.

    The built-in k-mer engine (used when BLAST+ is not installed) needs exact
    k-mer overlap, so the sequences are deliberately long.
    """
    long_seq = "ATCGATCGATCG" * 5
    query = tmp_path / "query.fasta"
    query.write_text(f">query\n{long_seq}\n", encoding="utf-8")
    database = tmp_path / "db.fasta"
    database.write_text(f">subject1\n{long_seq}\n>subject2\nGCTAGCTAGCTAGCTAGCTA\n",
                        encoding="utf-8")
    return query, database


def test_quick_blast_finds_the_subject_that_contains_the_query(blast_files):
    result = quick_blast(str(blast_files[0]), str(blast_files[1]))

    assert result.num_hits >= 1


def test_quick_blast_against_itself_reports_a_perfect_hit(tmp_path):
    long_seq = "ACGTACGTACGT" * 5
    fasta = tmp_path / "self.fasta"
    fasta.write_text(f">seq1\n{long_seq}\n", encoding="utf-8")

    result = quick_blast(str(fasta), str(fasta))

    assert result.num_hits >= 1
    assert result.top_hits(1)[0].percent_identity == pytest.approx(100.0)


def test_quick_blast_result_carries_the_fields_the_notebook_prints(blast_files):
    result = quick_blast(str(blast_files[0]), str(blast_files[1]))

    hit = result.top_hits(5)[0]
    assert hasattr(result, "num_hits")
    assert hit.query_id and hit.subject_id
    assert 0.0 <= hit.percent_identity <= 100.0
    assert hit.alignment_length > 0
