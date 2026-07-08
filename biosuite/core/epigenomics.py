"""
Epigenomics analysis: methylation, ATAC-seq, chromatin states.

Pure Python implementations using numpy/pandas/scipy.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as sp_stats


@dataclass
class MethylationSite:
    chrom: str
    pos: int
    context: str  # CpG, CHG, CHH
    methylation_level: float
    coverage: int
    methylated_count: int


@dataclass
class EpigenomicsReport:
    total_sites: int = 0
    avg_methylation: float = 0.0
    cpg_methylation: float = 0.0
    chg_methylation: float = 0.0
    chh_methylation: float = 0.0
    dmr_count: int = 0
    dmrs: list = field(default_factory=list)
    methylation_distribution: dict = field(default_factory=dict)
    message: str = ""


def parse_bisulfite_bed(bed_file):
    """Parse bisulfite sequencing BED file with methylation calls.

    Expected format: chrom, start, end, methylated_count, total_count, context
    """
    sites = []
    with open(bed_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            try:
                sites.append(MethylationSite(
                    chrom=parts[0],
                    pos=int(parts[1]),
                    context=parts[5] if len(parts) > 5 else 'CpG',
                    methylation_level=int(parts[3]) / max(int(parts[4]), 1),
                    coverage=int(parts[4]),
                    methylated_count=int(parts[3])
                ))
            except (ValueError, IndexError):
                continue
    return sites


def calculate_methylation_levels(sites, min_coverage=5):
    """Calculate methylation statistics from bisulfite sites.

    Args:
        sites: list of MethylationSite objects.
        min_coverage: minimum coverage to include site.

    Returns:
        EpigenomicsReport.
    """
    filtered = [s for s in sites if s.coverage >= min_coverage]
    if not filtered:
        return EpigenomicsReport(message="No sites passed coverage filter")

    levels = [s.methylation_level for s in filtered]
    contexts = {}
    for s in filtered:
        ctx = s.context
        if ctx not in contexts:
            contexts[ctx] = []
        contexts[ctx].append(s.methylation_level)

    report = EpigenomicsReport(
        total_sites=len(filtered),
        avg_methylation=np.mean(levels),
        cpg_methylation=np.mean(contexts.get('CpG', [0])),
        chg_methylation=np.mean(contexts.get('CHG', [0])),
        chh_methylation=np.mean(contexts.get('CHH', [0])),
        methylation_distribution={
            'unmethylated (< 20%)': sum(1 for l in levels if l < 0.2),
            'low (20-50%)': sum(1 for l in levels if 0.2 <= l < 0.5),
            'medium (50-80%)': sum(1 for l in levels if 0.5 <= l < 0.8),
            'high (> 80%)': sum(1 for l in levels if l >= 0.8),
        }
    )
    return report


def find_dmrs(sites_group1, sites_group2, min_coverage=5, p_threshold=0.05, min_delta=0.2):
    """Find differentially methylated regions between two groups.

    Args:
        sites_group1: list of MethylationSite for group 1.
        sites_group2: list of MethylationSite for group 2.
        min_coverage: minimum coverage.
        p_threshold: significance threshold.
        min_delta: minimum methylation difference.

    Returns:
        List of DMR dicts with chrom, pos, delta, p_value.
    """
    g1_map = {(s.chrom, s.pos): s for s in sites_group1 if s.coverage >= min_coverage}
    g2_map = {(s.chrom, s.pos): s for s in sites_group2 if s.coverage >= min_coverage}
    common = set(g1_map.keys()) & set(g2_map.keys())

    dmrs = []
    for key in sorted(common):
        s1 = g1_map[key]
        s2 = g2_map[key]
        delta = s1.methylation_level - s2.methylation_level

        if abs(delta) < min_delta:
            continue

        x1 = np.array([1] * s1.methylated_count + [0] * (s1.coverage - s1.methylated_count))
        x2 = np.array([1] * s2.methylated_count + [0] * (s2.coverage - s2.methylated_count))
        _, pval = sp_stats.mannwhitneyu(x1, x2, alternative='two-sided')

        if pval < p_threshold:
            dmrs.append({
                'chrom': key[0], 'pos': key[1],
                'delta_methylation': round(delta, 4),
                'p_value': round(pval, 6),
                'group1_level': round(s1.methylation_level, 4),
                'group2_level': round(s2.methylation_level, 4),
            })

    return dmrs


def parse_atac_peaks(peak_file):
    """Parse ATAC-seq peak BED file."""
    peaks = []
    with open(peak_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                peaks.append({
                    'chrom': parts[0],
                    'start': int(parts[1]),
                    'end': int(parts[2]),
                    'score': float(parts[4]) if len(parts) > 4 else 0,
                    'name': parts[3] if len(parts) > 3 else ''
                })
    return peaks


def atac_peak_stats(peaks):
    if not peaks:
        return {}
    lengths = [p['end'] - p['start'] for p in peaks]
    return {
        'total_peaks': len(peaks),
        'total_bp': sum(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'chromosomes': len(set(p['chrom'] for p in peaks)),
    }


def format_epigenomics_report(report):
    lines = [
        "=== Epigenomics Report ===",
        f"Total sites: {report.total_sites}",
        f"Average methylation: {report.avg_methylation:.3f}",
        f"CpG methylation: {report.cpg_methylation:.3f}",
        f"CHG methylation: {report.chg_methylation:.3f}",
        f"CHH methylation: {report.chh_methylation:.3f}",
        f"DMRs found: {report.dmr_count}",
    ]
    if report.methylation_distribution:
        lines.append("\nMethylation distribution:")
        for cat, count in report.methylation_distribution.items():
            lines.append(f"  {cat}: {count} sites ({count/max(report.total_sites,1)*100:.1f}%)")
    return '\n'.join(lines)
