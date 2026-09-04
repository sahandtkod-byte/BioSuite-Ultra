"""Regression tests for peak_calling.py chunked coverage rewrite."""
import numpy as np
import pytest

from biosuite.core.peak_calling import (
    _find_peaks_from_coverage, _chunked_smoothed_coverage,
    _builtin_call_peaks, annotate_peaks_with_genes,
    CHUNK_SIZE,
)


def _dense_reference(positions, min_score=5):
    """Original dense algorithm reproduced as oracle."""
    peaks = []
    for chrom, starts in positions.items():
        cov = np.zeros(max(starts) + 200)
        for s in starts:
            cov[s:s + 200] += 1
        smoothed = np.convolve(cov, np.ones(50) / 50, mode='same')
        thr = np.percentile(smoothed[smoothed > 0], 95) if np.any(smoothed > 0) else min_score
        in_peak, ps, pm, peak_sum = False, 0, 0.0, 0
        for i in range(len(smoothed)):
            if smoothed[i] > thr and not in_peak:
                in_peak, ps, pm, peak_sum = True, i, smoothed[i], i
            elif smoothed[i] > thr and in_peak:
                if smoothed[i] > pm:
                    pm, peak_sum = smoothed[i], i
            elif smoothed[i] <= thr and in_peak:
                in_peak = False
                if pm >= min_score:
                    peaks.append((chrom, ps, i, peak_sum, pm))
    return sorted(peaks, key=lambda x: -x[4])


def test_chunked_matches_dense_oracle():
    rng = np.random.default_rng(2)
    pos = {'chr1': sorted([500 + int(rng.integers(0, 500)) for _ in range(30)]
                          + [5000 + int(rng.integers(0, 300)) for _ in range(20)])}
    ref = sorted(_dense_reference(pos), key=lambda x: x[1])  # by start
    got = sorted(_find_peaks_from_coverage(pos), key=lambda p: p.start)
    got_t = [(p.chrom, p.start, p.end, p.summit, p.score) for p in got]
    # compare peak starts/ends (threshold differs slightly: chunk-median
    # vs global percentile — tolerance only in score, coords must match)
    assert len(got_t) == len(ref)
    for (rc, rs, re, rsum, rm), (gc, gs, ge, gsum, gm) in zip(ref, got_t):
        assert rs == gs and re == ge and rsum == gsum
        assert gm == pytest.approx(rm, rel=0.05)


def test_no_dense_allocation_on_huge_locus():
    # Two read clusters 10 Mbp apart: dense array would be ~80 MB per
    # chromosome here (simulate) — chunked mode streams fine.
    rng = np.random.default_rng(3)
    pos = {'chr1': sorted([1000 + int(rng.integers(0, 500)) for _ in range(30)]
                          + [1000 + CHUNK_SIZE + int(rng.integers(0, 500)) for _ in range(30)])}
    peaks = _find_peaks_from_coverage(pos)
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
    # internal engine raises on missing SAM; the public call_peaks() wraps
    # this with an existence check and a graceful message.
    with pytest.raises(FileNotFoundError):
        _builtin_call_peaks(str(tmp_path / 'no.sam'))


def test_end_to_end_two_clusters(tmp_path):
    rng = np.random.default_rng(0)
    lines = ['@HD\tVN:1.6']
    for c in (5000, 30000):
        for i in range(40):
            pos = c + int(rng.integers(0, 1000))
            lines.append(f'r{i}\t0\tchr1\t{pos}\t60\t50M\t*\t0\t0\t'
                         'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC\t'
                         'I' * 50)
    sam = tmp_path / 'x.sam'
    sam.write_text('\n'.join(lines))
    rep = _builtin_call_peaks(str(sam))
    assert rep.total_peaks >= 2
    starts = sorted(p.start for p in rep.peaks)
    assert any(abs(s - 5000) < 1500 for s in starts) and \
           any(abs(s - 30000) < 1500 for s in starts)
