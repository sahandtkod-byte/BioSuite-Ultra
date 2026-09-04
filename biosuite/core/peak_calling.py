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


CHUNK_SIZE = 5_000_000  # bp per coverage chunk (dense per-chromosome was
#                          a ~2 GB array on human chromosomes -> OOM)


def _chunked_smoothed_coverage(starts, window=200, chunk=CHUNK_SIZE):
    """Yield smoothed coverage arrays over consecutive genomic chunks.

    Dense per-chromosome coverage allocation made the built-in caller
    allocate ~2 GB for a single human chromosome and crash with a
    MemoryError; chunks cap memory at ~200 MB regardless of chromosome
    length.  Overlap of ``window`` bases between chunks keeps the running
    mean and the peak state machine seamless at chunk boundaries.
    """
    if not starts:
        return
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    max_pos = int(starts[-1]) + window
    kernel = np.ones(50) / 50
    offset = 0
    while offset < max_pos:
        end = min(offset + chunk, max_pos)
        padded_end = min(end + window, max_pos)
        lo = np.searchsorted(starts, offset)
        hi = np.searchsorted(starts, padded_end, side='right')
        cov = np.zeros(padded_end - offset)
        for s in starts[lo:hi]:
            cov[s - offset:min(s - offset + window, len(cov))] += 1
        smoothed = np.convolve(cov, kernel, mode='same')
        yield offset, smoothed[:end - offset]
        offset = end


def _find_peaks_from_coverage(positions, min_score=5, min_distance=200):
    """Peak-calling directly from read-start positions (chunked)."""
    peaks = []
    for chrom, starts in positions.items():
        if not starts:
            continue

        # First pass: pooled threshold over sampled chunks
        sample_vals = []
        for _off, smoothed in _chunked_smoothed_coverage(starts):
            pos = smoothed[smoothed > 0]
            if pos.size:
                sample_vals.append(np.percentile(pos, 95))
        threshold = float(np.median(sample_vals)) if sample_vals else float(min_score)

        # Second pass: state machine, seamless across chunks
        in_peak = False
        peak_start = 0
        peak_max = 0.0
        peak_summit = 0
        for off, smoothed in _chunked_smoothed_coverage(starts):
            for i in range(len(smoothed)):
                if smoothed[i] > threshold:
                    if not in_peak:
                        in_peak, peak_start = True, off + i
                        peak_max, peak_summit = smoothed[i], off + i
                    elif smoothed[i] > peak_max:
                        peak_max, peak_summit = smoothed[i], off + i
                elif in_peak:
                    in_peak = False
                    if peak_max >= min_score:
                        peaks.append(Peak(
                            chrom=chrom, start=peak_start, end=off + i,
                            summit=peak_summit, score=peak_max,
                            p_value=1e-5,
                            fold_enrichment=peak_max / max(threshold, 1.0),
                            length=off + i - peak_start
                        ))
        if in_peak and peak_max >= min_score:
            end = int(np.max(starts)) + 200
            peaks.append(Peak(
                chrom=chrom, start=peak_start, end=end, summit=peak_summit,
                score=peak_max, p_value=1e-5,
                fold_enrichment=peak_max / max(threshold, 1.0),
                length=end - peak_start
            ))

    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks


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
