"""
Epigenomics analysis: methylation, ATAC-seq, chromatin states.

Pure Python implementations using numpy/pandas/scipy.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from .log import get_logger

logger = get_logger(__name__)


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
    #: NaN when the dataset holds no site of that context — a mean over no sites
    #: is not 0.0, which would claim the context is unmethylated.
    cpg_methylation: float = float('nan')
    chg_methylation: float = float('nan')
    chh_methylation: float = float('nan')
    #: None until a caller has actually run find_dmrs() and recorded how many
    #: regions it returned; 0 would read as "no DMRs found".
    dmr_count: int | None = None
    dmrs: list = field(default_factory=list)
    methylation_distribution: dict = field(default_factory=dict)
    message: str = ""


def parse_bisulfite_bed(bed_file):
    """Parse bisulfite sequencing BED file with methylation calls.

    Expected format: chrom, start, end, methylated_count, total_count, context

    A row whose methylated count exceeds its total count is corrupt and is
    skipped with a warning: ``find_dmrs`` hands that pair to Fisher's exact test,
    which raises "All values in `table` must be nonnegative" and aborts the whole
    scan over one bad line.
    """
    sites = []
    with open(bed_file, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            try:
                methylated = int(parts[3])
                coverage = int(parts[4])
                if methylated > coverage:
                    logger.warning(
                        "Skipping %s:%s — %d methylated reads of %d total",
                        parts[0], parts[1], methylated, coverage)
                    continue
                sites.append(MethylationSite(
                    chrom=parts[0],
                    pos=int(parts[1]),
                    context=parts[5],
                    methylation_level=methylated / max(coverage, 1),
                    coverage=coverage,
                    methylated_count=methylated
                ))
            except (ValueError, IndexError):
                logger.warning("Skipping unreadable bisulfite BED line %d", number)
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

    def context_mean(name):
        """Mean over the sites of one context, or NaN when it has none.

        ``np.mean(contexts.get(name, [0]))` invented a zero: a CHH-only dataset
        reported 'CpG methylation: 0.000' and 'CHG methylation: 0.000', claiming
        both contexts are unmethylated when no site of either was measured.
        """
        values = contexts.get(name)
        return float(np.mean(values)) if values else float('nan')

    report = EpigenomicsReport(
        total_sites=len(filtered),
        avg_methylation=float(np.mean(levels)),
        cpg_methylation=context_mean('CpG'),
        chg_methylation=context_mean('CHG'),
        chh_methylation=context_mean('CHH'),
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

        # 2x2 contingency (methylated vs unmethylated, group 1 vs 2):
        # Fisher's exact test is the standard — the old Mann-Whitney over
        # tie-heavy binary vectors is inaccurate, especially at low counts.
        _, pval = sp_stats.fisher_exact([
            [s1.methylated_count, s1.coverage - s1.methylated_count],
            [s2.methylated_count, s2.coverage - s2.methylated_count],
        ])

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
    """Parse ATAC-seq peak BED file.

    A row whose coordinates or score are not numbers is skipped rather than
    raised: one commented header or a UCSC ``track`` line used to abort the whole
    file, which is not how the sibling bisulfite reader behaves.
    """
    peaks = []
    with open(peak_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                peaks.append({
                    'chrom': parts[0],
                    'start': int(parts[1]),
                    'end': int(parts[2]),
                    'score': float(parts[4]) if len(parts) > 4 else 0,
                    'name': parts[3] if len(parts) > 3 else ''
                })
            except ValueError:
                continue
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
    def level(value):
        """Render a context mean, or say that the context was never measured."""
        if value is None or np.isnan(value):
            return 'n/a (no site of this context)'
        return f"{value:.3f}"

    lines = [
        "=== Epigenomics Report ===",
        f"Total sites: {report.total_sites}",
        f"Average methylation: {report.avg_methylation:.3f}",
        f"CpG methylation: {level(report.cpg_methylation)}",
        f"CHG methylation: {level(report.chg_methylation)}",
        f"CHH methylation: {level(report.chh_methylation)}",
        f"DMRs found: {'not assessed' if report.dmr_count is None else report.dmr_count}",
    ]
    if report.methylation_distribution:
        lines.append("\nMethylation distribution:")
        for cat, count in report.methylation_distribution.items():
            lines.append(f"  {cat}: {count} sites ({count/max(report.total_sites,1)*100:.1f}%)")
    return '\n'.join(lines)
