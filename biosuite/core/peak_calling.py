"""
ChIP-seq peak calling with dual-mode execution.

Pure Python pileup-based peak caller as default, MACS2 as optional.
"""
import os
import subprocess
import tempfile
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field


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


def _compute_coverage(positions, window=200, step=50):
    coverage = {}
    for chrom, starts in positions.items():
        if not starts:
            continue
        max_pos = max(starts) + window
        cov = np.zeros(max_pos)
        for s in starts:
            end = min(s + window, max_pos)
            cov[s:end] += 1
        coverage[chrom] = cov
    return coverage


def _find_peaks_from_coverage(coverage, min_score=5, min_distance=200):
    peaks = []
    for chrom, cov in coverage.items():
        smoothed = np.convolve(cov, np.ones(50) / 50, mode='same')
        threshold = np.percentile(smoothed[smoothed > 0], 95) if np.any(smoothed > 0) else min_score

        in_peak = False
        peak_start = 0
        peak_max = 0
        peak_summit = 0

        for i in range(len(smoothed)):
            if smoothed[i] > threshold and not in_peak:
                in_peak = True
                peak_start = i
                peak_max = smoothed[i]
                peak_summit = i
            elif smoothed[i] > threshold and in_peak:
                if smoothed[i] > peak_max:
                    peak_max = smoothed[i]
                    peak_summit = i
            elif smoothed[i] <= threshold and in_peak:
                in_peak = False
                if peak_max >= min_score:
                    peaks.append(Peak(
                        chrom=chrom, start=peak_start, end=i,
                        summit=peak_summit, score=peak_max,
                        p_value=1e-5, fold_enrichment=peak_max / max(threshold, 1),
                        length=i - peak_start
                    ))

    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks


def _builtin_call_peaks(sam_file, output_bed=None, min_score=5):
    positions = _read_sam_positions(sam_file)
    if not positions:
        return PeakReport(engine='builtin', message="No mapped reads found.")

    coverage = _compute_coverage(positions)
    peaks = _find_peaks_from_coverage(coverage, min_score=min_score)

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
                dist = min(abs(peak.start - g['start']), abs(peak.end - g['end']))
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
