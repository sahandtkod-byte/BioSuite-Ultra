"""The built-in peak caller's statistics, against independent computations.

The coverage model is a MACS-style Poisson on a smoothed pileup, and every piece of
it is testable without a genome: the chunked coverage must be *bit-identical* to
computing the whole chromosome at once (that was the original MemoryError fix), the
background lambda must be the mean of that coverage, the p-value must be the upper
tail of a Poisson computed here from its own pmf sum, and the FDR correction must
agree with a reference implementation written as a plain loop.  A SAM reader is
tested on the four record kinds it has to skip.
"""
import math

import numpy as np
import pytest

from biosuite.core import peak_calling as pc


def sam_line(name, flag=0, chrom="chr1", pos=100, mapq=60, seq="ACGT", cig="4M"):
    return "\t".join(str(field) for field in
                     (name, flag, chrom, pos, mapq, cig, "*", 0, 0, seq, "IIII")) + "\n"


@pytest.fixture
def sam_file(tmp_path):
    def write(*lines):
        path = tmp_path / "reads.sam"
        path.write_text("@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:100000" + "\n" + "".join(lines),
                        encoding="utf-8")
        return str(path)

    return write


# ─── the SAM reader's four refusals ──────────────────────────────────────────

def test_the_tool_check_answers_for_macs2_only(monkeypatch):
    monkeypatch.setattr(pc, "_has_tool", lambda name: name == "macs2")

    assert pc.check_peak_tools() == {'macs2': True}


def test_reads_are_grouped_by_reference_and_headers_are_ignored(sam_file):
    positions = pc._read_sam_positions(sam_file(sam_line("r1", pos=100),
                                                 sam_line("r2", pos=140),
                                                 sam_line("r3", chrom="chr2", pos=10)))

    assert dict(positions) == {"chr1": [100, 140], "chr2": [10]}


@pytest.mark.parametrize("line, why", [
    (sam_line("unmapped", flag=4), "the unmapping flag"),
    (sam_line("secondary", flag=256), "the secondary-alignment flag"),
    (sam_line("lowq", mapq=5), "a MAPQ under the floor"),
    ("ragged\tonly\tnine\tfields\there\tfor\tnow\tplease", "too few columns"),
])
def test_a_read_that_cannot_carry_signal_is_left_out(sam_file, line, why):
    positions = pc._read_sam_positions(sam_file(line))

    assert dict(positions) == {}, f"{why} should not have been counted"


def test_the_mapq_floor_is_the_value_the_caller_set(sam_file):
    path = sam_file(sam_line("r1", mapq=20))

    assert dict(pc._read_sam_positions(path, min_mapq=20)) == {"chr1": [100]}
    assert dict(pc._read_sam_positions(path, min_mapq=21)) == {}, "20 is below 21"


# ─── coverage, computed in chunks ────────────────────────────────────────────

def test_no_reads_produce_no_chunks():
    assert list(pc._chunked_smoothed_coverage([])) == []


def test_a_chunked_coverage_is_the_same_as_one_whole_computation():
    """The padding around every chunk exists for this: cutting the chromosome must
    not change a single value of the pileup."""
    starts = [0, 120, 950, 1_000, 2_400, 3_900, 4_050]
    whole = [smoothed for _offset, smoothed in
             pc._chunked_smoothed_coverage(starts, window=200, chunk=10_000_000)]
    pieces = list(pc._chunked_smoothed_coverage(starts, window=200, chunk=1_000))

    joined = np.concatenate([smoothed for _offset, smoothed in pieces])
    assert sum(len(s) for _o, s in pieces) == len(joined)
    assert np.array_equal(joined, np.concatenate(whole)), \
        "chunk boundaries changed the coverage"


def test_a_read_spanning_the_chunk_boundary_is_counted_on_both_sides():
    starts = [950]                      # covers 950..1150 with window 200
    pieces = list(pc._chunked_smoothed_coverage(starts, window=200, chunk=1_000))

    first, second = pieces
    assert first[0] == 0 and second[0] == 1_000
    assert second[1][0] > 0.0, "the read that began in chunk 1 vanished in chunk 2"
    assert np.any(first[1][50:] > 0.0), "and it was counted where it began"


def test_a_region_with_no_reads_at_all_yields_an_empty_chunk_not_a_crash():
    pieces = list(pc._chunked_smoothed_coverage([0, 100_000], window=200, chunk=20_000))

    assert [offset for offset, _ in pieces] == [0, 20_000, 40_000, 60_000, 80_000, 100_000]
    assert np.any(pieces[2][1] == 0.0), "the middle of the gap is zero coverage"
    assert len(pieces[2][1]) == 20_000


# ─── the Poisson model ───────────────────────────────────────────────────────

def test_the_background_lambda_is_the_mean_depth_of_the_covered_span():
    assert pc._background_lambda([]) == 0.0
    assert pc._background_lambda([0], window=200) == pytest.approx(1.0)
    assert pc._background_lambda([0, 400], window=200) == pytest.approx(2 * 200 / 600)
    assert pc._background_lambda([7, 7, 7], window=200) == pytest.approx(3.0), \
        "three reads stacked on one spot give three times the window over one window"


def test_an_expected_depth_of_zero_still_answers_the_probability_question():
    assert pc._poisson_p_value(5, 0.0) == 0.0, "no background makes any pileup extreme"
    assert pc._poisson_p_value(0, 0.0) == 1.0, "and observing nothing at all is certain"


def _upper_tail(lam, k):
    """P(X >= k) for a Poisson(lam), summed from a recursively built pmf.

    Written here on purpose: the module under test gets it from scipy, and a
    factorial-based sum overflows long before it reaches 400 terms.
    """
    pmf = math.exp(-lam)
    total = 0.0
    i = 0
    while True:
        if i >= k:
            total += pmf
            if pmf < 1e-18:
                return total
        i += 1
        pmf *= lam / i


@pytest.mark.parametrize("observed, lam", [(5, 2.0), (1, 0.5), (40, 12.5)])
def test_the_p_value_is_the_upper_tail_of_a_Poisson(observed, lam):
    assert pc._poisson_p_value(observed, lam) == pytest.approx(
        _upper_tail(lam, math.ceil(observed)), rel=1e-6)


def test_the_threshold_and_the_p_value_answer_the_same_question():
    """The peak scan keeps a position when its depth is *strictly* above the
    threshold; the p-value each peak carries must call the same depths
    significant, or a reported peak's own q-value contradicts why it was kept."""
    lam, cutoff = 4.0, 0.01
    cut = pc._poisson_threshold(lam, cutoff, floor=2.0)

    by_p = [d for d in range(1, 25) if pc._poisson_p_value(d, lam) <= cutoff]
    by_depth = [d for d in range(1, 25) if d > cut]

    assert by_p == by_depth, (by_p, by_depth)
    assert cut >= 2.0


def test_a_zero_background_thresholds_at_the_floor():
    assert pc._poisson_threshold(0.0, 0.01, floor=3.0) == 3.0
    assert pc._poisson_threshold(0.0, 0.01, floor=0.0) == 1.0


def test_an_impossible_cutoff_falls_back_to_the_background_rather_than_infinity():
    """``isf(0)`` is infinite; a score of infinity would reject every position."""
    depth = pc._poisson_threshold(6.0, 0.0, floor=1.0)

    assert np.isfinite(depth)
    assert depth == pytest.approx(6.0)


# ─── FDR and run detection ───────────────────────────────────────────────────

def _bh_reference(p_values):
    """The Benjamini-Hochberg step-up correction, written as the textbook loop."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * n / rank)
        q[index] = running
    return q


@pytest.mark.parametrize("p_values", [
    [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.08, 0.086, 0.127, 0.128, 0.152],
    [0.5, 0.5, 0.5],
    [1e-12, 0.9, 0.04],
    [0.02],
])
def test_the_fdr_correction_matches_a_reference_implementation(p_values):
    q = pc._benjamini_hochberg(p_values)

    assert np.allclose(q, _bh_reference(p_values), atol=1e-12), (q, p_values)
    assert np.all(q >= np.array(p_values) - 1e-15), "a q-value never undershoots its p"
    assert np.all(q <= 1.0)


def test_no_p_values_produce_no_q_values():
    q = pc._benjamini_hochberg([])

    assert q.size == 0 and q.dtype == float


@pytest.mark.parametrize("mask, runs", [
    ([True, True, False, True], [(0, 2), (3, 4)]),
    ([False, False], []),
    ([True], [(0, 1)]),
    ([False, True, True, False, True], [(1, 3), (4, 5)]),
])
def test_runs_are_reported_as_half_open_index_pairs(mask, runs):
    found = pc._runs_above(np.array(mask, dtype=bool))

    assert [tuple(pair) for pair in found] == runs


def test_an_empty_mask_has_no_runs_but_still_answers_with_the_right_shape():
    found = pc._runs_above(np.array([], dtype=bool))

    assert found.shape == (0, 2)
