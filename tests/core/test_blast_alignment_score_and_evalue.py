"""The built-in engine's two numbers: the banded alignment score and the E-value estimate.

`_banded_align_score` returns one 7-tuple — `(score, matches, mismatches, q_start,
t_start, q_end, t_end)` — and `_builtin_search` unpacks exactly seven, so the shape is
part of the contract: the no-room guard used to return six, and the failure surfaced far
away as `ValueError: not enough values to unpack`.  Both scorers are pinned on exact
values, and the relation between them (`matches + mismatches` is the aligned column
count, and for a gapless alignment that is also both spans) is pinned too, because
`alignment_length` and `percent_identity` are computed from it.
"""
import json

import pytest

from biosuite.core import blast as bl


@pytest.mark.parametrize("seq1, seq2, expected", [
    pytest.param("ACGTACGT", "ACGTACGT", (8.0, 8, 0, 0, 0, 8, 8), id="perfect-match"),
    pytest.param("ACGTACGT", "ACGTTCGA", (5.0, 6, 1, 0, 0, 7, 7), id="two-changes"),
    pytest.param("AAAACCCCCAAA", "AAAACCCCCTAAA", (10.0, 11, 1, 0, 0, 12, 12),
                 id="one-mismatch-inside"),
    pytest.param("ACGT", "TTTT", (1.0, 1, 0, 3, 0, 4, 1),
                 id="no-shared-context-but-one-column"),
])
def test_the_scorer_returns_score_matches_mismatches_and_both_spans(seq1, seq2, expected):
    assert bl._banded_align_score(seq1, seq2, 0, 0) == expected


@pytest.mark.parametrize("seq1, seq2", [
    pytest.param("ACGTACGT", "ACGTACGT", id="perfect"),
    pytest.param("ACGTACGT", "ACGTTCGA", id="two-changes"),
    pytest.param("AAAACCCCCAAA", "AAAACCCCCTAAA", id="one-mismatch-inside"),
])
def test_matches_plus_mismatches_is_the_aligned_column_count_of_both_spans(seq1, seq2):
    """`_builtin_search` builds `alignment_length` and the identity percentage from
    these two numbers, so the three must describe the same columns."""
    _score, matches, mismatches, q_start, t_start, q_end, t_end = \
        bl._banded_align_score(seq1, seq2, 0, 0)
    columns = matches + mismatches
    assert columns == q_end - q_start == t_end - t_start
    assert matches <= columns and mismatches >= 0


def test_the_scorer_offers_no_alignment_when_the_anchors_have_no_room():
    """`start1` past the end leaves nothing to extend.  Seven values, zero columns:
    the caller's `columns <= 0` test is what drops the candidate."""
    assert bl._banded_align_score("ACGT", "ACGT", 4, 4) == (0, 0, 0, 4, 4, 4, 4)
    assert len(bl._banded_align_score("ACGT", "ACGT", 3, 0)) == 7


def test_a_single_mismatching_column_falls_back_to_the_naive_scorer():
    """Biopython's local aligner returns no HSP for `T` vs `A` (a mismatch costs more
    than no alignment at all), and the fallback answers -1/0/1 for the same window.
    Pinned because the two scorers disagree by construction and the caller cannot tell
    them apart from the numbers alone."""
    assert bl._banded_align_score("ACGT", "ACGT", 3, 0) == (-1, 0, 1, 3, 0, 4, 1)


def test_a_local_alignment_stops_at_the_divergence_instead_of_scoring_through_it():
    """A 160-column identical head followed by 400 mismatching columns scores the head,
    not the window: `mode='local'` and mismatch = -1 mean extending would only lower
    the score."""
    head = "ACGT" * 40                                  # 160 columns, identical
    score, matches, mismatches, qs, ts, qe, te = \
        bl._banded_align_score(head + "A" * 400, head + "C" * 400, 0, 0)
    assert (score, matches, mismatches) == (160.0, 160, 0)
    assert (qs, ts, qe, te) == (0, 0, 160, 160)


def test_only_the_first_500_columns_of_a_window_are_ever_scored():
    """`max_len = min(len1, len2, 500)` caps the band: a 600-column identical pair is
    scored over 500, and the same pair anchored at 100 covers 100..600."""
    assert bl._banded_align_score("A" * 600, "A" * 600, 0, 0) == (500.0, 500, 0, 0, 0, 500, 500)
    assert bl._banded_align_score("A" * 600, "A" * 600, 100, 100) == \
        (500.0, 500, 0, 100, 100, 600, 600)


# ── E-value estimate ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("score, expected", [
    pytest.param(5, 25146.36100264203, id="weak-hit-loses"),
    pytest.param(20, 0.0004449576153739951, id="modest-hit"),
    pytest.param(100, 2.0113329430913902e-45, id="strong-hit"),
])
def test_the_evalue_estimator_follows_its_own_closed_form(score, expected):
    """`K · db_size · (query_len - k + 1) · exp(-λ·score)` with λ=1.19, K=0.025 over a
    1 Mb database and a 400-base query."""
    assert bl._estimate_evalue(score, db_size=1_000_000, query_len=400) == \
        pytest.approx(expected)


def test_a_non_positive_score_is_reported_as_no_evidence_rather_than_a_huge_number():
    assert bl._estimate_evalue(0, 1_000_000, 400) == 1.0
    assert bl._estimate_evalue(-3, 100, 40) == 1.0


def test_the_estimate_grows_with_the_database_and_shrinks_with_the_score():
    small = bl._estimate_evalue(20, db_size=1_000, query_len=400)
    large = bl._estimate_evalue(20, db_size=1_000_000, query_len=400)
    assert small == pytest.approx(4.449576153739951e-07)
    assert large == pytest.approx(0.0004449576153739951)
    assert large / small == pytest.approx(1000.0), "the space is linear in the estimate"
    assert bl._estimate_evalue(25, 1_000_000, 400) == pytest.approx(1.1594885831159773e-06)
    assert bl._estimate_evalue(25, 1_000_000, 400) < large


def test_a_query_shorter_than_the_seed_still_has_one_window():
    """`max(query_len - k + 1, 1)` — without the floor a 5-base query against a k=15
    seed would multiply the estimate by zero and report E = 0.0 for every hit."""
    assert bl._estimate_evalue(20, 1_000_000, 5, k=15) == \
        pytest.approx(1.1527399362020597e-06)
    assert bl._estimate_evalue(20, 1_000_000, 400, k=5) == pytest.approx(0.00045648501473601565)


def test_the_estimator_returns_a_plain_float():
    """`np.exp` hands back `numpy.float64`, and a hit whose `e_value` is a numpy scalar
    still prints — but `json.dumps` of the whole hit used to raise
    `TypeError: Object of type int64 is not JSON serializable` because the coordinates
    came out of Biopython as numpy integers."""
    value = bl._estimate_evalue(20, 1_000_000, 400)
    assert type(value) is float


def test_a_builtin_hit_is_json_serialisable_without_a_default(tmp_path):
    query = tmp_path / "q.fa"
    query.write_text(">q\nAAAAAAAAAAACGTACGTACGTACGTACGTGGGGG\n", encoding="utf-8")
    subject = tmp_path / "d.fa"
    subject.write_text(">s\nTTTTTTTTTTGTACGTACGTACGTACGACCCCC\n", encoding="utf-8")

    hit = bl._builtin_search(str(query), str(subject), evalue=1.0).hits[0]

    assert (hit.query_start, hit.query_end, hit.subject_start, hit.subject_end,
            hit.alignment_length, hit.mismatches) == (17, 27, 11, 21, 11, 0)
    payload = {name: getattr(hit, name) for name in hit.__dataclass_fields__}
    assert json.loads(json.dumps(payload)) == payload, "round-trips through JSON"
    assert all(type(value) in (str, int, float) for value in payload.values())
