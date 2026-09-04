"""
ChIP-seq peak calling with dual-mode execution.

Pure Python pileup-based peak caller as default, MACS2 as optional.
"""
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson


@dataclass
class Peak:
    chrom: str
    start: int
    end: int
    summit: int
    score: float
    p_value: float
    fold_enrichment: float
    length: int = 0
    name: str = ""
    #: Benjamini-Hochberg FDR across the peaks called in the same run.
    q_value: float = float('nan')
    #: Poisson background rate (lambda) the p-value was computed against.
    background: float = float('nan')
    #: Pileup depth a region had to exceed to be considered.
    threshold: float = float('nan')


@dataclass
class PeakReport:
    engine: str
    total_peaks: int = 0
    peaks: list = field(default_factory=list)
    output_bed: str = ""
    message: str = ""


from .utils import has_tool as _has_tool


def check_peak_tools():
    return {'macs2': _has_tool('macs2')}


def _read_sam_positions(sam_file, min_mapq=20):
    positions = defaultdict(list)
    with open(sam_file) as f:
        for line in f:
            if line.startswith('@'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            flag = int(parts[1])
            if flag & 4 or flag & 256:
                continue
            mapq = int(parts[4])
            if mapq < min_mapq:
                continue
            positions[parts[2]].append(int(parts[3]))
    return positions


SMOOTH_WINDOW = 50   # bases of moving-average smoothing applied to the pileup

CHUNK_SIZE = 5_000_000  # bp per coverage chunk (dense per-chromosome was
#                          a ~2 GB array on human chromosomes -> OOM)


def _chunked_smoothed_coverage(starts, window=200, chunk=CHUNK_SIZE):
    """Yield ``(offset, smoothed)`` coverage arrays over consecutive chunks.

    Dense per-chromosome coverage allocation made the built-in caller
    allocate ~2 GB for a single human chromosome and crash with a
    MemoryError; chunks cap memory at ~200 MB regardless of chromosome
    length.

    Each chunk is computed with ``window + SMOOTH_WINDOW`` bases of context on
    *both* sides and then trimmed, so the result is bit-for-bit identical to
    computing the whole chromosome at once: reads that start before a chunk
    but extend into it are counted, and the smoothing kernel never sees a
    fabricated zero edge in the interior of the chromosome.

    Coverage is built from a difference array (O(reads)) instead of a Python
    loop that touched every covered base (O(reads x window)).
    """
    if len(starts) == 0:
        return
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    max_pos = int(starts[-1]) + window
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    pad = window + SMOOTH_WINDOW
    offset = 0
    while offset < max_pos:
        end = min(offset + chunk, max_pos)
        lo_bound = max(0, offset - pad)
        hi_bound = min(max_pos, end + pad)
        # Reads whose [start, start+window) interval intersects the padded
        # region.  searchsorted on the left bound must reach back a full
        # window, otherwise reads spanning the chunk boundary are lost.
        lo = int(np.searchsorted(starts, lo_bound - window, side='left'))
        hi = int(np.searchsorted(starts, hi_bound, side='right'))
        span = hi_bound - lo_bound
        diff = np.zeros(span + 1)
        if hi > lo:
            rel_start = np.clip(starts[lo:hi] - lo_bound, 0, span)
            rel_end = np.clip(starts[lo:hi] + window - lo_bound, 0, span)
            np.add.at(diff, rel_start, 1.0)
            np.add.at(diff, rel_end, -1.0)
        cov = np.cumsum(diff[:-1])
        smoothed = np.convolve(cov, kernel, mode='same')
        left = offset - lo_bound
        yield offset, smoothed[left:left + (end - offset)]
        offset = end


def _background_lambda(starts, window=200):
    """Mean smoothed coverage across the covered span = Poisson background.

    This is the ``lambda_bg`` of a MACS-style model: the expected pileup depth
    if the same number of reads were distributed uniformly over the region
    that actually carries signal.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return 0.0
    span = int(starts.max()) - int(starts.min()) + window
    if span <= 0:
        return float(starts.size)
    # Total coverage mass = reads x window (each read covers `window` bases).
    return float(starts.size * window) / float(span)


def _poisson_p_value(observed, lam):
    """Upper-tail Poisson probability of seeing at least *observed* depth.

    Note:
        The pileup is smoothed over ``SMOOTH_WINDOW`` bases before this test is
        applied, so neighbouring positions are correlated and the reported
        p-value is an approximation in exactly the same sense as MACS's; it is
        a ranking statistic, not an exact significance level.  It is computed
        from the data, never hard-coded.
    """
    if lam <= 0:
        return 0.0 if observed > 0 else 1.0
    k = int(np.ceil(observed))
    return float(poisson.sf(k - 1, lam))


def _poisson_threshold(lam, p_cutoff, floor):
    """Smallest depth whose Poisson p-value is below *p_cutoff*."""
    if lam <= 0:
        return float(max(floor, 1.0))
    # isf gives the value k with sf(k) <= p_cutoff.
    k = float(poisson.isf(p_cutoff, lam))
    if not np.isfinite(k):
        k = lam
    return float(max(k, floor, lam))


def _benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR q-values for a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _runs_above(mask):
    """Return ``(start, stop)`` index pairs of True runs in a boolean array."""
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return np.stack([starts, stops], axis=1)


def _find_peaks_from_coverage(positions, min_score=5, min_distance=200,
                              p_cutoff=1e-5, threshold=None, window=200):
    """Call peaks from read-start positions using a Poisson background model.

    Args:
        positions: mapping of chromosome -> list of read start coordinates.
        min_score: minimum smoothed pileup depth for a region to be reported.
        min_distance: minimum distance between summits of reported peaks;
            closer peaks are merged into the stronger one.
        p_cutoff: Poisson upper-tail probability used to derive the enrichment
            threshold.
        threshold: explicit pileup threshold; overrides the Poisson-derived
            one (used by tests and by callers that want a fixed cut).
        window: assumed read footprint in bases.

    Returns:
        list[Peak]: peaks sorted by score, each carrying a **computed**
        Poisson p-value and a Benjamini-Hochberg q-value.  Enrichment is
        expressed over the Poisson background, not over a quantile of the
        sample's own signal.
    """
    peaks = []
    for chrom, starts in positions.items():
        if len(starts) == 0:
            continue

        lam = _background_lambda(starts, window=window)
        cut = (float(threshold) if threshold is not None
               else _poisson_threshold(lam, p_cutoff, float(min_score)))

        # Vectorised scan: locate runs above the threshold inside each chunk
        # and stitch runs that straddle a chunk boundary.  The previous
        # implementation stepped through every base of the genome in Python.
        pending = None      # (start, max, summit) of a run open at chunk end
        for off, smoothed in _chunked_smoothed_coverage(starts, window=window):
            if smoothed.size == 0:
                continue
            mask = smoothed > cut
            # A run left open by the previous chunk ends exactly at this
            # chunk's first base unless the enrichment continues into it.
            if pending is not None and not mask[0]:
                peaks.append(_make_peak(chrom, pending, off, lam,
                                        min_score, cut))
                pending = None
            runs = _runs_above(mask)
            for lo, hi in runs:
                segment = smoothed[lo:hi]
                local_arg = int(np.argmax(segment))
                seg_max = float(segment[local_arg])
                seg_summit = off + lo + local_arg
                if pending is not None and lo == 0:
                    # Continues the run that straddled the chunk border.
                    p_start, p_max, p_summit = pending
                    if seg_max > p_max:
                        p_max, p_summit = seg_max, seg_summit
                    pending = (p_start, p_max, p_summit)
                else:
                    pending = (int(off + lo), seg_max, int(seg_summit))
                if hi < mask.size:
                    # The run is fully contained in this chunk.
                    peaks.append(_make_peak(chrom, pending, int(off + hi),
                                            lam, min_score, cut))
                    pending = None
        if pending is not None:
            end = int(np.max(starts)) + window
            peaks.append(_make_peak(chrom, pending, end, lam, min_score, cut))

    peaks = [p for p in peaks if p is not None]
    peaks = _merge_close_peaks(peaks, min_distance)
    if peaks:
        q_values = _benjamini_hochberg([p.p_value for p in peaks])
        for peak, q in zip(peaks, q_values):
            peak.q_value = float(q)
    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks


def _make_peak(chrom, pending, end, lam, min_score, threshold):
    """Build a Peak from an open run, or None when it is below min_score."""
    start, peak_max, summit = pending
    if peak_max < min_score:
        return None
    return Peak(
        chrom=chrom, start=int(start), end=int(end), summit=int(summit),
        score=float(peak_max),
        p_value=_poisson_p_value(peak_max, lam),
        fold_enrichment=float(peak_max / lam) if lam > 0 else float('inf'),
        length=int(end - start),
        background=float(lam),
        threshold=float(threshold),
    )


def _merge_close_peaks(peaks, min_distance):
    """Collapse peaks whose summits are closer than *min_distance*.

    ``min_distance`` was accepted but ignored, so a single broad enrichment
    could be reported as several adjacent "independent" peaks.
    """
    if min_distance <= 0 or len(peaks) < 2:
        return list(peaks)
    merged = []
    for peak in sorted(peaks, key=lambda p: (p.chrom, p.start)):
        if merged and merged[-1].chrom == peak.chrom and \
                peak.summit - merged[-1].summit < min_distance:
            prev = merged[-1]
            if peak.score > prev.score:
                prev.score = peak.score
                prev.summit = peak.summit
                prev.p_value = peak.p_value
                prev.fold_enrichment = peak.fold_enrichment
            prev.end = max(prev.end, peak.end)
            prev.length = prev.end - prev.start
        else:
            merged.append(peak)
    return merged


def _builtin_call_peaks(sam_file, output_bed=None, min_score=5):
    positions = _read_sam_positions(sam_file)
    if not positions:
        return PeakReport(engine='builtin', message="No mapped reads found.")

    peaks = _find_peaks_from_coverage(positions, min_score=min_score)

    report = PeakReport(
        engine='builtin',
        total_peaks=len(peaks),
        peaks=peaks,
        message=f"Built-in peak caller: {len(peaks)} peaks found"
    )

    if output_bed:
        _write_bed(peaks, output_bed)
        report.output_bed = output_bed

    return report


def _write_bed(peaks, output_file):
    with open(output_file, 'w') as f:
        for p in peaks:
            f.write(f"{p.chrom}\t{p.start}\t{p.end}\t{p.name}\t{p.score}\t.\n")


# ── MACS2 Wrapper ───────────────────────────────────────────────────────────

def _macs2_call_peaks(bam_file, output_dir, name='chipseq', genome='hs'):
    cmd = ['macs2', 'callpeak', '-t', bam_file, '-f', 'BAM',
           '-g', genome, '--outdir', output_dir, '-n', name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ── Public API ──────────────────────────────────────────────────────────────

def call_peaks(input_file, output_bed=None, genome='hs', tool='auto'):
    if not os.path.exists(input_file):
        return PeakReport(engine='none', message=f"File not found: {input_file}")

    tools = check_peak_tools()

    if tool in ('macs2', 'auto') and tools['macs2']:
        out_dir = tempfile.mkdtemp()
        if _macs2_call_peaks(input_file, out_dir):
            return PeakReport(engine='macs2', output_bed=out_dir,
                             message="Using MACS2 (external)")

    if output_bed is None:
        output_bed = tempfile.mktemp(suffix='.bed')
    return _builtin_call_peaks(input_file, output_bed)


def annotate_peaks_with_genes(peaks, gene_bed_file):
    genes = []
    with open(gene_bed_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                genes.append({'chrom': parts[0], 'start': int(parts[1]),
                             'end': int(parts[2]), 'name': parts[3]})

    for peak in peaks:
        nearest = None
        min_dist = float('inf')
        for g in genes:
            if g['chrom'] == peak.chrom:
                # distance 0 for overlapping intervals (was computed even
                # then, so intersected genes were artificially distant)
                if peak.start < g['end'] and g['start'] < peak.end:
                    dist = 0
                else:
                    dist = min(abs(peak.start - g['end']), abs(g['start'] - peak.end))
                if dist < min_dist:
                    min_dist = dist
                    nearest = g['name']
        peak.name = nearest or "intergenic"

    return peaks


def format_peak_report(report):
    lines = [
        "=== Peak Calling Report ===",
        f"Engine: {report.engine}",
        f"Total peaks: {report.total_peaks}",
    ]
    if report.peaks:
        scores = [p.score for p in report.peaks]
        lines.append(f"Score range: {min(scores):.1f} — {max(scores):.1f}")
    if report.message:
        lines.append(f"Note: {report.message}")
    return '\n'.join(lines)
