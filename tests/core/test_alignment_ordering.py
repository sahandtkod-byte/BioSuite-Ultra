"""Row-order and input-handling regression tests for MSA (NEW-15, NEW-16).

Both the built-in progressive aligner and the external tool wrappers returned
alignment rows in guide-tree order, so row i was not input sequence i.  Callers
doing ``zip(names, alignment.sequences)`` mislabelled sequences, and the order
additionally depended on whether MUSCLE or MAFFT happened to be installed.

The defect was hidden because the previous test sorted both sides before
comparing, which discards exactly the property that was broken.
"""
import pytest

from biosuite.core.msa import auto_align


def _ungapped(result):
    return [seq.replace("-", "") for _, seq in result.sequences]


def _names(result):
    return [name for name, _ in result.sequences]


# ── row order ───────────────────────────────────────────────────────────────

def test_row_order_matches_input_order():
    seqs = ["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC", "GGGGTTTTCC", "ACGTACGTAG"]
    assert _ungapped(auto_align(seqs)) == seqs


def test_row_order_holds_when_the_first_input_is_the_outlier():
    """Guide-tree order puts the outlier last; input order keeps it first."""
    seqs = ["TTTTTTTTTTTT", "ACGTACGTACGT", "ACGTACGTACGA", "ACGTACGTACGC"]
    assert _ungapped(auto_align(seqs)) == seqs


def test_names_stay_paired_with_their_own_sequence():
    named = [("alpha", "ACGTACGTAA"), ("beta", "TTGGCCAA"),
             ("gamma", "ACGTACGTAC"), ("delta", "GGGGTTTTCC")]
    result = auto_align(named)
    assert _names(result) == ["alpha", "beta", "gamma", "delta"]
    for (name, original), (rname, aligned) in zip(named, result.sequences):
        assert rname == name
        assert aligned.replace("-", "") == original, f"{name} carries another row"


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_order_preserved_for_various_input_sizes(n):
    seqs = [f"ACGT{'A' * i}CGTACGT" for i in range(n)]
    assert _ungapped(auto_align(seqs)) == seqs


def test_duplicate_sequences_do_not_collapse_rows():
    seqs = ["ACGTACGT", "ACGTACGT", "TTTTGGGG"]
    result = auto_align(seqs)
    assert len(result.sequences) == 3
    assert _ungapped(result) == seqs


def test_order_is_stable_across_repeated_calls():
    seqs = ["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC", "GGGGTTTTCC"]
    first = _ungapped(auto_align(seqs))
    for _ in range(5):
        assert _ungapped(auto_align(seqs)) == first


# ── structural invariants ───────────────────────────────────────────────────

def test_all_rows_have_equal_length():
    seqs = ["ACGTACGTAA", "TTGGCC", "ACGTACGTACGTAC", "GGGG"]
    result = auto_align(seqs)
    assert len({len(seq) for _, seq in result.sequences}) == 1


def test_alignment_length_field_matches_the_rows():
    result = auto_align(["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC"])
    assert all(len(seq) == result.alignment_length for _, seq in result.sequences)


def test_conservation_length_matches_the_alignment():
    """BSU-011: conservation used to come back empty."""
    result = auto_align(["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC"])
    assert len(result.conservation) == result.alignment_length
    assert result.conservation
    assert all(0.0 <= c <= 1.0 for c in result.conservation)


def test_removing_gaps_recovers_the_input_exactly():
    seqs = ["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC", "GGGGTTTTCC"]
    result = auto_align(seqs)
    for original, (_, aligned) in zip(seqs, result.sequences):
        assert aligned.replace("-", "") == original


def test_num_sequences_matches_the_input():
    seqs = ["ACGTACGTAA", "TTGGCCAA", "ACGTACGTAC"]
    assert auto_align(seqs).num_sequences == 3


# ── input handling (NEW-16, BSU-011) ────────────────────────────────────────

def test_a_bare_string_is_rejected():
    """A FASTA *path* used to be 'aligned' character by character."""
    with pytest.raises(TypeError):
        auto_align("/tmp/sequences.fasta")


def test_a_bare_sequence_string_is_rejected():
    with pytest.raises(TypeError):
        auto_align("ACGTACGTACGT")


def test_fewer_than_two_sequences_is_an_explicit_no_op():
    """BSU-011: the input used to be discarded silently."""
    result = auto_align(["ACGTACGT"])
    assert result.num_sequences == 1
    assert result.sequences[0][1] == "ACGTACGT"
    assert "2" in result.message or "least" in result.message.lower()


def test_empty_input_is_handled():
    result = auto_align([])
    assert result.num_sequences == 0
    assert result.sequences == []


def test_single_sequence_keeps_its_content():
    result = auto_align([("only", "ACGTACGT")])
    assert result.sequences == [("only", "ACGTACGT")]


def test_two_sequences_align():
    result = auto_align(["ACGTACGT", "ACGTTCGT"])
    assert result.num_sequences == 2
    assert _ungapped(result) == ["ACGTACGT", "ACGTTCGT"]
