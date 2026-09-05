"""Regression tests for the built-in ChIP-seq peak caller.

Covers three separate defect classes:

* **Chunking must be exact.**  The chunked coverage builder is compared against
  a dense, single-array reference implementation written independently in this
  file.  The old version dropped reads that started before a chunk but
  extended into it, which corrupted the pileup at every chunk boundary.
* **Statistics must be computed, not asserted.**  Every peak used to be
  emitted with a hard-coded ``p_value=1e-5`` and the detection threshold was
  the 95th percentile of the sample's *own* signal, so uniform background
  noise produced a long list of "peaks".
* **The scan must be exact.**  The vectorised run finder is compared against a
  literal base-by-base state machine.
"""
import functools

import numpy as np
import pytest
from scipy.stats import poisson

from biosuite.core.peak_calling import (
    _find_peaks_from_coverage, _chunked_smoothed_coverage,
    _builtin_call_peaks, annotate_peaks_with_genes,
    SMOOTH_WINDOW, CHUNK_SIZE,
)


# ── independent oracles ─────────────────────────────────────────────────────

def _dense_smoothed(starts, window=200):
    """Reference pileup: one dense array, no chunking, no tricks."""
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    max_pos = int(starts[-1]) + window
    cov = np.zeros(max_pos)
    for s in starts:
        cov[s:min(s + window, max_pos)] += 1
    return np.convolve(cov, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='same')


def _dense_runs(smoothed, cut):
    """Reference scan: literal base-by-base state machine."""
    runs, in_peak, start, best, summit = [], False, 0, 0.0, 0
    for i, v in enumerate(smoothed):
        if v > cut:
            if not in_peak:
                in_peak, start, best, summit = True, i, v, i
            elif v > best:
                best, summit = v, i
        elif in_peak:
            in_peak = False
            runs.append((start, i, summit, best))
    if in_peak:
        runs.append((start, len(smoothed), summit, best))
    return runs


# ── coverage exactness ──────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk", [997, 4096, 10_000, CHUNK_SIZE])
def test_chunked_coverage_is_bit_exact_against_dense_reference(chunk):
    rng = np.random.default_rng(7)
    starts = sorted(int(x) for x in rng.integers(0, 40_000, 900))
    expected = _dense_smoothed(starts)
    got = np.concatenate(
        [seg for _, seg in _chunked_smoothed_coverage(starts, chunk=chunk)])
    assert got.size == expected.size
    # Chunking is an implementation detail: it must not change a single value.
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_reads_spanning_a_chunk_boundary_are_counted():
    """A read starting just before a boundary still covers bases after it."""
    chunk = 1000
    starts = [chunk - 10]          # covers [990, 1190): crosses the boundary
    got = np.concatenate(
        [seg for _, seg in _chunked_smoothed_coverage(starts, chunk=chunk)])
    expected = _dense_smoothed(starts)
    assert np.allclose(got, expected, rtol=0, atol=1e-12)
    assert got[1100] > 0          # would have been 0 with the boundary bug


def test_numpy_array_input_is_accepted():
    """`if not starts:` raised ValueError for ndarray input."""
    starts = np.array([100, 200, 300])
    segments = list(_chunked_smoothed_coverage(starts))
    assert segments and segments[0][1].size > 0


# ── scan exactness ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk", [997, 4096, CHUNK_SIZE])
def test_vectorised_scan_matches_base_by_base_state_machine(chunk, monkeypatch):
    rng = np.random.default_rng(7)
    starts = sorted(int(x) for x in rng.integers(0, 40_000, 900))
    reference = _dense_smoothed(starts)
    cut = float(np.percentile(reference[reference > 0], 99))
    expected = sorted((s, e, summit, round(float(best), 9))
                      for s, e, summit, best in _dense_runs(reference, cut))

    monkeypatch.setattr(
        'biosuite.core.peak_calling._chunked_smoothed_coverage',
        functools.partial(_chunked_smoothed_coverage, chunk=chunk))
    peaks = _find_peaks_from_coverage({'chr1': starts}, min_score=0,
                                      min_distance=0, threshold=cut)
    got = sorted((p.start, p.end, p.summit, round(p.score, 9)) for p in peaks)
    assert got == expected


# ── statistics ──────────────────────────────────────────────────────────────

def test_uniform_background_yields_no_peaks():
    """Random uniform reads contain no enrichment, so no peaks may be called.

    The old percentile-of-own-signal threshold reported 115 "peaks" here, all
    with the same fabricated p-value.
    """
    rng = np.random.default_rng(11)
    starts = sorted(int(x) for x in rng.integers(0, 200_000, 4000))
    peaks = _find_peaks_from_coverage({'chr1': starts})
    assert peaks == []


def test_spiked_loci_are_recovered_with_computed_statistics():
    rng = np.random.default_rng(11)
    background = [int(x) for x in rng.integers(0, 200_000, 4000)]
    truth = (50_000, 150_000)
    starts = sorted(background
                    + [truth[0] + int(x) for x in rng.integers(0, 300, 400)]
                    + [truth[1] + int(x) for x in rng.integers(0, 300, 400)])
    peaks = _find_peaks_from_coverage({'chr1': starts})

    assert len(peaks) == 2
    summits = sorted(p.summit for p in peaks)
    for expected, summit in zip(truth, summits):
        assert abs(summit - expected) < 500

    for peak in peaks:
        # The p-value must be reproducible from the reported background with
        # an independent implementation of the Poisson tail...
        independent = poisson.sf(int(np.ceil(peak.score)) - 1, peak.background)
        assert peak.p_value == pytest.approx(independent, abs=1e-15)
        # ...and must not be the old hard-coded constant.
        assert peak.p_value != 1e-5
        assert 0.0 <= peak.p_value <= 1.0
        assert peak.background > 0
        assert peak.fold_enrichment == pytest.approx(
            peak.score / peak.background)
        assert 0.0 <= peak.q_value <= 1.0


def test_p_values_track_signal_strength():
    """A stronger locus must not be reported as less significant."""
    rng = np.random.default_rng(3)
    background = [int(x) for x in rng.integers(0, 200_000, 4000)]
    starts = sorted(background
                    + [40_000 + int(x) for x in rng.integers(0, 300, 120)]
                    + [140_000 + int(x) for x in rng.integers(0, 300, 600)])
    peaks = sorted(_find_peaks_from_coverage({'chr1': starts}),
                   key=lambda p: p.score)
    assert len(peaks) == 2
    weak, strong = peaks
    assert strong.score > weak.score
    assert strong.p_value <= weak.p_value
    assert strong.fold_enrichment > weak.fold_enrichment


def test_close_summits_are_merged():
    """min_distance was accepted but ignored."""
    rng = np.random.default_rng(5)
    background = [int(x) for x in rng.integers(0, 100_000, 2000)]
    starts = sorted(background
                    + [30_000 + int(x) for x in rng.integers(0, 60, 200)]
                    + [30_100 + int(x) for x in rng.integers(0, 60, 200)])
    merged = _find_peaks_from_coverage({'chr1': starts}, min_distance=2000)
    split = _find_peaks_from_coverage({'chr1': starts}, min_distance=0)
    assert len(merged) <= len(split)
    assert len(merged) == 1


# ── unchanged behaviour that must keep working ──────────────────────────────

def test_no_dense_allocation_on_huge_locus():
    rng = np.random.default_rng(3)
    pos = {'chr1': sorted([1000 + int(rng.integers(0, 500)) for _ in range(60)]
                          + [1000 + CHUNK_SIZE + int(rng.integers(0, 500))
                             for _ in range(60)])}
    peaks = _find_peaks_from_coverage(pos, min_score=1)
    assert any(p.start < 10_000 for p in peaks)
    assert any(p.start > CHUNK_SIZE for p in peaks)


def test_annotate_overlap_distance_zero(tmp_path):
    from biosuite.core.peak_calling import Peak
    genes = tmp_path / "g.bed"
    genes.write_text("chr1\t900\t2000\tgeneA\nchr1\t5000\t6000\tgeneB\n")
    peaks = [Peak(chrom='chr1', start=1000, end=1500, summit=1200,
                  score=10, p_value=1e-5, fold_enrichment=5)]
    out = annotate_peaks_with_genes(peaks, str(genes))
    assert out[0].name == 'geneA'


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _builtin_call_peaks(str(tmp_path / 'no.sam'))


def test_end_to_end_two_clusters(tmp_path):
    rng = np.random.default_rng(0)
    lines = ['@HD\tVN:1.6']
    for cluster in (5000, 30000):
        for i in range(80):
            pos = cluster + int(rng.integers(0, 400))
            lines.append(f'r{cluster}_{i}\t0\tchr1\t{pos}\t60\t50M\t*\t0\t0\t'
                         'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC\t'
                         'I' * 50)
    sam = tmp_path / 'x.sam'
    sam.write_text('\n'.join(lines))
    rep = _builtin_call_peaks(str(sam))
    assert rep.total_peaks >= 2
    starts = sorted(p.start for p in rep.peaks)
    assert any(abs(s - 5000) < 1500 for s in starts)
    assert any(abs(s - 30000) < 1500 for s in starts)
    assert all(p.p_value != 1e-5 for p in rep.peaks)
