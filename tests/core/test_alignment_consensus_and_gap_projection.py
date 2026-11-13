"""Alignment consensus and gap projection — the two helpers that build the
profile consensus line.

`_consensus` picks the most frequent non-gap character per column and `_project_gaps`
re-inserts the gaps a consensus alignment introduced into a sequence that had
none.  Both had only ever run on the happy path, so the empty input and the
"original is shorter than the alignment" case — the two that produce a silently
wrong consensus rather than an error — were untested.
"""
import pytest

from biosuite.core.alignment import _consensus, _project_gaps


# ── _consensus ───────────────────────────────────────────────────────────────

def test_no_sequences_produce_an_empty_consensus():
    assert _consensus([]) == ''


def test_a_column_of_only_gaps_is_kept_as_a_gap():
    assert _consensus(['--', 'AT']) == 'AT'
    assert _consensus(['--', '--']) == '--'


def test_the_majority_base_wins_and_a_tie_keeps_the_first_maximum():
    """max() on a dict returns the first key reaching the highest count, so a tie
    resolves by insertion order — pinned because a different rule would change
    the consensus of real alignments."""
    assert _consensus(['AA', 'CC', 'AA']) == 'AA'
    assert _consensus(['AC', 'CA']) == 'AC'


def test_gaps_do_not_compete_with_bases():
    assert _consensus(['A-', '-A', 'AC']) == 'AA'


def test_a_ragged_input_only_scores_the_positions_it_has():
    """The loop guards with `pos < len(s)`, so a short sequence contributes to the
    early columns only instead of raising IndexError."""
    assert _consensus(['ATGC', 'AT']) == 'ATGC'


def test_consensus_of_a_single_sequence_is_that_sequence():
    assert _consensus(['AC-GT']) == 'AC-GT'


# ── _project_gaps ────────────────────────────────────────────────────────────

def test_gaps_from_the_alignment_are_inserted_at_their_own_positions():
    assert _project_gaps('ACGT', 'A-C-GT') == 'A-C-GT'


def test_an_original_shorter_than_the_alignment_gets_gaps_not_a_crash():
    """Once the original runs out, every remaining alignment column — gap or base
    — becomes a gap: there is no character left to place there."""
    assert _project_gaps('AC', 'A-C-G-T') == 'A-C----'
    assert _project_gaps('', '--') == '--'


def test_extra_original_characters_are_appended():
    """The alignment being shorter than the sequence must not silently drop the
    tail of the original."""
    assert _project_gaps('ACGTT', 'AC-G') == 'AC-GTT'


def test_projection_preserves_the_original_bases_exactly():
    original = 'MKVLA'
    aligned = 'MK-VLA'
    assert _project_gaps(original, aligned) == aligned
    assert [c for c in _project_gaps(original, aligned) if c != '-'] == list(original)


@pytest.mark.parametrize("aligned, expected", [
    ("", "AC"),
    ("-", "-AC"),
    ("--", "--AC"),
])
def test_gap_columns_before_the_sequence_come_first_then_the_tail(aligned, expected):
    """With fewer alignment columns than residues, the columns are emitted in
    order and what remains of the original is appended after them."""
    assert _project_gaps("AC", aligned) == expected
