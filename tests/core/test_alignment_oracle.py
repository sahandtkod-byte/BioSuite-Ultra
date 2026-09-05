"""Independent verification of the pairwise sequence aligners (BSU-001).

The point of this module is that **nothing here calls the implementation to
decide what the right answer is**.  A textbook scalar dynamic-programming
routine is written from scratch below and used as the oracle, alongside
invariants that any correct alignment must satisfy.

Historical defect: the vectorised DP filled row ``i`` by reading the
horizontal predecessor from the *previous* row, so gap extensions inside a row
were scored against stale values.  Scores were silently too low on inputs
containing gaps.
"""
import random

import pytest

from biosuite.core.alignment import needleman_wunsch, smith_waterman


def _nw(a, b):
    """needleman_wunsch returns (aligned1, aligned2, score)."""
    aligned1, aligned2, score = needleman_wunsch(a, b, MATCH, MISMATCH, GAP)
    return aligned1, aligned2, score


def _sw(a, b):
    aligned1, aligned2, score = smith_waterman(a, b, MATCH, MISMATCH, GAP)
    return aligned1, aligned2, score

MATCH, MISMATCH, GAP = 2, -1, -2


# ── oracle: plain textbook DP, no numpy, no vectorisation ───────────────────

def _nw_score(a, b, match=MATCH, mismatch=MISMATCH, gap=GAP):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            dp[i][j] = max(diag, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
    return dp[n][m]


def _sw_score(a, b, match=MATCH, mismatch=MISMATCH, gap=GAP):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            dp[i][j] = max(0, diag, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            best = max(best, dp[i][j])
    return best


def _rescore(aligned_a, aligned_b, match=MATCH, mismatch=MISMATCH, gap=GAP):
    """Score a produced alignment directly from the gapped strings."""
    total = 0
    for x, y in zip(aligned_a, aligned_b):
        if x == '-' or y == '-':
            total += gap
        elif x == y:
            total += match
        else:
            total += mismatch
    return total


# ── known answers computed by hand ──────────────────────────────────────────

@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0),
    ("A", "A", MATCH),
    ("A", "C", MISMATCH),
    ("A", "", GAP),
    ("", "AC", 2 * GAP),
    ("ACGT", "ACGT", 4 * MATCH),
    # One deletion in the middle: 4 matches + 1 gap.
    ("ACGTT", "ACTT", 4 * MATCH + GAP),
    # Two-base internal insertion: 4 matches + 2 gaps.
    ("ACGTACGT", "ACGTGGACGT", 8 * MATCH + 2 * GAP),
])
def test_needleman_wunsch_known_answers(a, b, expected):
    assert _nw(a, b)[2] == expected


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0),
    ("AAAA", "TTTT", 0),                    # no positively scoring local hit
    ("GGGACGTGGG", "TTTACGTTTT", 4 * MATCH),
    ("ACGT", "ACGT", 4 * MATCH),
])
def test_smith_waterman_known_answers(a, b, expected):
    assert _sw(a, b)[2] == expected


# ── differential test against the scalar oracle ─────────────────────────────

def _random_pairs(n, alphabet="ACGT", max_len=18, seed=1234):
    rng = random.Random(seed)
    for _ in range(n):
        a = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))
        b = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))
        yield a, b


def test_needleman_wunsch_matches_scalar_dp_on_random_input():
    mismatches = []
    for a, b in _random_pairs(400):
        got = _nw(a, b)[2]
        want = _nw_score(a, b)
        if got != want:
            mismatches.append((a, b, got, want))
    assert not mismatches, f"{len(mismatches)} disagreements, e.g. {mismatches[:3]}"


def test_smith_waterman_matches_scalar_dp_on_random_input():
    mismatches = []
    for a, b in _random_pairs(400, seed=99):
        got = _sw(a, b)[2]
        want = _sw_score(a, b)
        if got != want:
            mismatches.append((a, b, got, want))
    assert not mismatches, f"{len(mismatches)} disagreements, e.g. {mismatches[:3]}"


def test_exhaustive_short_binary_alphabet():
    """All 2-letter strings up to length 4 x length 4, both algorithms."""
    from itertools import product
    words = ['']
    for length in range(1, 5):
        words += [''.join(w) for w in product("AC", repeat=length)]
    for a in words:
        for b in words:
            assert _nw(a, b)[2] == _nw_score(a, b)
            assert _sw(a, b)[2] == _sw_score(a, b)


# ── invariants that hold for any correct alignment ──────────────────────────

def test_returned_alignment_reproduces_the_reported_score():
    for a, b in _random_pairs(150, seed=7):
        aligned1, aligned2, score = _nw(a, b)
        assert _rescore(aligned1, aligned2) == score


def test_returned_alignment_preserves_the_input_sequences():
    for a, b in _random_pairs(150, seed=8):
        aligned1, aligned2, _ = _nw(a, b)
        assert aligned1.replace('-', '') == a
        assert aligned2.replace('-', '') == b
        assert len(aligned1) == len(aligned2)


def test_local_score_never_exceeds_a_perfect_self_alignment():
    for a, b in _random_pairs(120, seed=11):
        local = _sw(a, b)[2]
        assert 0 <= local <= MATCH * min(len(a), len(b))


def test_alignment_is_symmetric_in_its_arguments():
    for a, b in _random_pairs(120, seed=13):
        assert _nw(a, b)[2] == _nw(b, a)[2]
        assert _sw(a, b)[2] == _sw(b, a)[2]


def test_gap_penalty_is_actually_applied_inside_a_row():
    """Direct guard for the stale-previous-row defect.

    A long internal insertion forces many consecutive horizontal moves within
    one DP row, which is exactly the case the broken vectorised fill scored
    against values from the previous row.
    """
    a = "ACGT" + "ACGT"
    b = "ACGT" + "TTTTTTTTTT" + "ACGT"
    assert _nw(a, b)[2] == _nw_score(a, b)
