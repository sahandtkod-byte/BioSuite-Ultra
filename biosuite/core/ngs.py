"""
Next Generation Sequencing utilities: BAM/SAM, coverage, VCF.

Provides functions for reading SAM/BAM files, computing per-base and
whole-genome coverage, intersecting BED and VCF intervals, and
computing VCF-level summary statistics (including Ts/Tv ratio).

All functions are pure-Python where possible; optional heavy deps
(pysam, scipy) are guarded with HAS_* flags.
"""
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

def read_bam_header(bam_path):
    """Return header dictionary from BAM file."""
    if not HAS_PYSAM:
        print("pysam not installed. pip install pysam")
        return None
    if not os.path.exists(bam_path):
        print(f"BAM file not found: {bam_path}")
        return None
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            return bam.header
    except Exception as e:
        print(f"Error reading BAM: {e}")
        return None



def read_vcf(vcf_path, max_variants=None, chunk_size=10000):
    """Read VCF file and return DataFrame with CHROM, POS, REF, ALT, QUAL, INFO.

    Args:
        vcf_path: path to VCF file.
        max_variants: maximum number of variants to read (None for all).
        chunk_size: for large files, read in chunks to limit memory.

    Returns:
        DataFrame with variant data, or None on error.
    """
    if not os.path.exists(vcf_path):
        print(f"VCF file not found: {vcf_path}")
        return None
    columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    try:
        data = []
        count = 0
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    data.append(parts[:8])
                    count += 1
                    if max_variants and count >= max_variants:
                        break
                    # Warn for large files
                    if count == 100000:
                        print(f"Warning: VCF has 100K+ variants. Consider using max_variants parameter.")
        if not data:
            print("No variant data found in VCF file.")
            return None
        df = pd.DataFrame(data, columns=columns)
        df['POS'] = df['POS'].astype(int)
        df['QUAL'] = pd.to_numeric(df['QUAL'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading VCF: {e}")
        return None

def manhattan_from_vcf(vcf_df, pvalue_col='QUAL', threshold=5e-8):
    """Convert VCF DataFrame to Manhattan plot data (chrom, pos, -log10(p))."""
    df = vcf_df.copy()
    df['chrom_num'] = df['CHROM'].str.replace('chr', '', case=False)
    df = df.sort_values(['chrom_num', 'POS'])
    pvals = pd.to_numeric(df['QUAL'], errors='coerce').fillna(0).clip(lower=0)
    df['neg_log10'] = -np.log10(pvals + 1e-300)
    return df[['CHROM', 'POS', 'neg_log10']]


# ── SAM / BAM Parsing ────────────────────────────────────────────────────────

def read_sam(sam_path):
    """Read a SAM file and return a list of parsed read records.

    Pure-Python parser that handles plain-text SAM (not BAM).  For BAM
    files, use ``read_bam_header`` and ``compute_coverage`` with pysam.

    Args:
        sam_path: Path to a SAM file (plain text, may be gzip-compressed
                  but not BAM binary).

    Returns:
        List of dicts, each representing one aligned read with keys:
        QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN,
        SEQ, QUAL, and any optional TAG fields.  Returns ``None`` on
        error or missing file.
    """
    if not os.path.exists(sam_path):
        print(f"SAM file not found: {sam_path}")
        return None
    try:
        reads = []
        with open(sam_path, 'r') as fh:
            for line in fh:
                if line.startswith('@'):
                    continue  # skip header
                fields = line.strip().split('\t')
                if len(fields) < 11:
                    continue
                record = {
                    'QNAME': fields[0],
                    'FLAG': int(fields[1]),
                    'RNAME': fields[2],
                    'POS': int(fields[3]),
                    'MAPQ': int(fields[4]),
                    'CIGAR': fields[5],
                    'RNEXT': fields[6],
                    'PNEXT': int(fields[7]),
                    'TLEN': int(fields[8]),
                    'SEQ': fields[9],
                    'QUAL': fields[10],
                }
                # Parse optional tag fields (TAG:TYPE:VALUE)
                for tag_field in fields[11:]:
                    parts = tag_field.split(':', 2)
                    if len(parts) == 3:
                        record[parts[0]] = parts[2]
                reads.append(record)
        return reads
    except Exception as e:
        print(f"Error reading SAM: {e}")
        return None


def parse_sam_header(sam_path):
    """Parse header lines from a SAM file.

    Args:
        sam_path: Path to SAM file.

    Returns:
        Dict with keys '@HD', '@SQ' (list of dicts), '@RG' (list),
        '@PG' (list), and 'raw' (list of header strings).
    """
    if not os.path.exists(sam_path):
        print(f"SAM file not found: {sam_path}")
        return None
    header = {'@HD': {}, '@SQ': [], '@RG': [], '@PG': [], 'raw': []}
    try:
        with open(sam_path, 'r') as fh:
            for line in fh:
                if not line.startswith('@'):
                    break
                line = line.strip()
                header['raw'].append(line)
                tag = line[:3]
                fields = line.split('\t')
                if tag == '@HD':
                    for f in fields[1:]:
                        k, v = f.split(':', 1)
                        header['@HD'][k] = v
                elif tag == '@SQ':
                    sq = {}
                    for f in fields[1:]:
                        k, v = f.split(':', 1)
                        sq[k] = v
                    header['@SQ'].append(sq)
                elif tag == '@RG':
                    rg = {}
                    for f in fields[1:]:
                        k, v = f.split(':', 1)
                        rg[k] = v
                    header['@RG'].append(rg)
                elif tag == '@PG':
                    pg = {}
                    for f in fields[1:]:
                        k, v = f.split(':', 1)
                        pg[k] = v
                    header['@PG'].append(pg)
        return header
    except Exception as e:
        print(f"Error parsing SAM header: {e}")
        return None


def parse_sam_read(line):
    """Parse a single SAM alignment line into a dict.

    Args:
        line: A single non-header line from a SAM file.

    Returns:
        Dict with standard SAM fields plus optional tags, or ``None``
        if the line cannot be parsed.
    """
    fields = line.strip().split('\t')
    if len(fields) < 11:
        return None
    try:
        record = {
            'QNAME': fields[0],
            'FLAG': int(fields[1]),
            'RNAME': fields[2],
            'POS': int(fields[3]),
            'MAPQ': int(fields[4]),
            'CIGAR': fields[5],
            'RNEXT': fields[6],
            'PNEXT': int(fields[7]),
            'TLEN': int(fields[8]),
            'SEQ': fields[9],
            'QUAL': fields[10],
        }
        for tag_field in fields[11:]:
            parts = tag_field.split(':', 2)
            if len(parts) == 3:
                record[parts[0]] = parts[2]
        return record
    except (ValueError, IndexError):
        return None


# ── Whole-Genome Coverage ────────────────────────────────────────────────────

def compute_coverage(bam_path, region=None, window_size=None):
    """Compute per-base or windowed coverage from a BAM file.

    Args:
        bam_path: Path to BAM file.
        region: Optional region string (e.g. 'chr1:1000-2000').  If
            ``None``, computes whole-genome coverage.
        window_size: If set, aggregate coverage into non-overlapping
            windows of this size.  Returns a DataFrame with columns
            [chrom, start, end, mean_coverage].

    Returns:
        numpy array of per-base coverage values (or DataFrame if
        ``window_size`` is set).  Returns ``None`` on error.
    """
    if not HAS_PYSAM:
        print("pysam required for coverage. pip install pysam")
        return None
    if not os.path.exists(bam_path):
        print(f"BAM file not found: {bam_path}")
        return None
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            if region:
                cov = np.sum(bam.count_coverage(region=region), axis=0)
                if window_size and window_size > 1:
                    return _windowed_coverage_array(cov, window_size)
                return cov
            else:
                # Whole-genome: iterate over each reference
                results = {}
                for ref_name in bam.references:
                    ref_len = bam.get_reference_length(ref_name)
                    if ref_len == 0:
                        continue
                    # count_coverage returns tuple of 4 arrays (A,C,G,T)
                    cov = np.sum(bam.count_coverage(reference=ref_name), axis=0)
                    if window_size and window_size > 1:
                        results[ref_name] = _windowed_coverage_array(
                            cov, window_size, ref_name
                        )
                    else:
                        results[ref_name] = cov
                return results
    except Exception as e:
        print(f"Coverage error: {e}")
        return None


def _windowed_coverage_array(cov, window_size, chrom=None):
    """Aggregate a per-base coverage array into fixed-size windows.

    Args:
        cov: 1-D numpy coverage array.
        window_size: Window size in bases.
        chrom: Optional chromosome name for the output DataFrame.

    Returns:
        DataFrame with columns [start, end, mean_coverage] (and chrom
        if provided).
    """
    n_windows = len(cov) // window_size
    if n_windows == 0:
        return pd.DataFrame({
            'start': [0], 'end': [len(cov)],
            'mean_coverage': [float(np.mean(cov))]
        })
    trimmed = cov[:n_windows * window_size]
    reshaped = trimmed.reshape(n_windows, window_size)
    means = reshaped.mean(axis=1)
    starts = np.arange(n_windows) * window_size
    ends = starts + window_size
    df = pd.DataFrame({
        'start': starts, 'end': ends, 'mean_coverage': means
    })
    if chrom is not None:
        df.insert(0, 'chrom', chrom)
    return df


# ── BED / VCF Intersection ──────────────────────────────────────────────────

def _parse_bed(bed_path):
    """Parse a BED file into a list of (chrom, start, end) tuples.

    Handles BED3 through BED12; only the first three columns are used.
    """
    intervals = []
    with open(bed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                intervals.append((parts[0], int(parts[1]), int(parts[2])))
    return intervals


def _parse_vcf_intervals(vcf_path):
    """Parse a VCF file into a list of (chrom, start, end) tuples.

    Each variant record becomes a 1-base interval at POS.
    """
    intervals = []
    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom = parts[0]
                pos = int(parts[1])
                intervals.append((chrom, pos - 1, pos))  # 0-based half-open
    return intervals


def intersect_bed_vcf(bed_path, vcf_path, report='count'):
    """Intersect BED intervals with VCF variant positions.

    Implements a simple interval overlap check.  For each BED interval,
    counts how many VCF variants fall within it (or returns the list of
    overlapping variants).

    Args:
        bed_path: Path to BED file.
        vcf_path: Path to VCF file.
        report: 'count' returns per-interval overlap counts;
                'detail' returns a DataFrame with [bed_chrom, bed_start,
                bed_end, vcf_chrom, vcf_pos, vcf_ref, vcf_alt].

    Returns:
        Series (report='count') or DataFrame (report='detail'), or
        ``None`` on error.
    """
    if not os.path.exists(bed_path):
        print(f"BED file not found: {bed_path}")
        return None
    if not os.path.exists(vcf_path):
        print(f"VCF file not found: {vcf_path}")
        return None

    bed = _parse_bed(bed_path)
    vcf_intervals = _parse_vcf_intervals(vcf_path)

    # Build chrom -> sorted list of (start, end, extra) for VCF
    vcf_by_chrom = defaultdict(list)
    for chrom, start, end in vcf_intervals:
        vcf_by_chrom[chrom].append((start, end))

    for chrom in vcf_by_chrom:
        vcf_by_chrom[chrom].sort()

    if report == 'detail':
        rows = []
        for chrom, start, end in bed:
            for vs, ve in vcf_by_chrom.get(chrom, []):
                if vs < end and ve > start:
                    rows.append({
                        'bed_chrom': chrom, 'bed_start': start,
                        'bed_end': end, 'vcf_chrom': chrom,
                        'vcf_pos': vs + 1  # back to 1-based
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        counts = []
        for chrom, start, end in bed:
            cnt = 0
            for vs, ve in vcf_by_chrom.get(chrom, []):
                if vs < end and ve > start:
                    cnt += 1
            counts.append(cnt)
        return pd.Series(counts, name='overlap_count')


def coverage_from_bed(bam_path, bed_path):
    """Compute mean coverage over each BED interval.

    Uses pysam to compute average coverage within each interval.

    Args:
        bam_path: Path to BAM file.
        bed_path: Path to BED file.

    Returns:
        DataFrame with columns [chrom, start, end, mean_coverage,
        total_bases], or ``None`` on error.
    """
    if not HAS_PYSAM:
        print("pysam required for coverage_from_bed. pip install pysam")
        return None
    if not os.path.exists(bam_path):
        print(f"BAM file not found: {bam_path}")
        return None
    if not os.path.exists(bed_path):
        print(f"BED file not found: {bed_path}")
        return None

    bed = _parse_bed(bed_path)
    rows = []
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for chrom, start, end in bed:
                region = f"{chrom}:{start+1}-{end}"
                try:
                    cov = np.sum(
                        bam.count_coverage(region=region), axis=0
                    )
                    mean_cov = float(np.mean(cov)) if len(cov) > 0 else 0.0
                except Exception:
                    mean_cov = 0.0
                rows.append({
                    'chrom': chrom, 'start': start, 'end': end,
                    'mean_coverage': mean_cov,
                    'total_bases': end - start
                })
    except Exception as e:
        print(f"Error computing coverage: {e}")
        return None
    return pd.DataFrame(rows)


# ── VCF Statistics ───────────────────────────────────────────────────────────

def vcf_summary(vcf_path):
    """Compute summary statistics from a VCF file.

    Returns counts by type (SNP, indel, multi-allelic), per-chromosome
    counts, and basic quality statistics.

    Args:
        vcf_path: Path to VCF file.

    Returns:
        Dict with keys: 'total_variants', 'snp_count', 'indel_count',
        'multi_allelic_count', 'ti_count', 'tv_count', 'ti_tv_ratio',
        'per_chrom' (dict), 'qual_mean', 'qual_median',
        'qual_gt_20_count'.
    """
    if not os.path.exists(vcf_path):
        print(f"VCF file not found: {vcf_path}")
        return None

    ti_count = 0
    tv_count = 0
    snp_count = 0
    indel_count = 0
    multi_allelic_count = 0
    per_chrom = defaultdict(int)
    quals = []

    _purines = set('AG')
    _pyrimidines = set('CT')

    try:
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                chrom = parts[0]
                qual_str = parts[5]
                ref = parts[3]
                alt_field = parts[4]

                per_chrom[chrom] += 1

                # Parse quality
                try:
                    q = float(qual_str)
                    quals.append(q)
                except ValueError:
                    pass

                # Classify variant
                alts = alt_field.split(',')
                if len(alts) > 1:
                    multi_allelic_count += 1

                for alt in alts:
                    if len(ref) == 1 and len(alt) == 1:
                        snp_count += 1
                        # Ts/Tv
                        if (ref in _purines and alt in _purines) or \
                           (ref in _pyrimidines and alt in _pyrimidines):
                            ti_count += 1
                        else:
                            tv_count += 1
                    else:
                        indel_count += 1

        total = snp_count + indel_count
        ti_tv = ti_count / tv_count if tv_count > 0 else float('inf')
        quals_arr = np.array(quals) if quals else np.array([0.0])

        return {
            'total_variants': total,
            'snp_count': snp_count,
            'indel_count': indel_count,
            'multi_allelic_count': multi_allelic_count,
            'ti_count': ti_count,
            'tv_count': tv_count,
            'ti_tv_ratio': round(ti_tv, 4),
            'per_chrom': dict(per_chrom),
            'qual_mean': float(np.mean(quals_arr)),
            'qual_median': float(np.median(quals_arr)),
            'qual_gt_20_count': int(np.sum(quals_arr > 20)),
        }
    except Exception as e:
        print(f"VCF summary error: {e}")
        return None


def compute_ts_tv_ratio(vcf_path):
    """Compute transition/transversion ratio from a VCF file.

    A transition is a purine↔purine (A↔G) or pyrimidine↔pyrimidine
    (C↔T) substitution.  A transversion is any purine↔pyrimidine
    substitution.

    Args:
        vcf_path: Path to VCF file.

    Returns:
        Dict with 'transitions', 'transversions', 'ratio', or ``None``.
    """
    if not os.path.exists(vcf_path):
        print(f"VCF file not found: {vcf_path}")
        return None

    _purines = set('AG')
    _pyrimidines = set('CT')
    ti = 0
    tv = 0

    try:
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                ref = parts[3]
                alt_field = parts[4]
                for alt in alt_field.split(','):
                    if len(ref) == 1 and len(alt) == 1:
                        if (ref in _purines and alt in _purines) or \
                           (ref in _pyrimidines and alt in _pyrimidines):
                            ti += 1
                        else:
                            tv += 1
        ratio = ti / tv if tv > 0 else float('inf')
        return {
            'transitions': ti,
            'transversions': tv,
            'ratio': round(ratio, 4)
        }
    except Exception as e:
        print(f"Ts/Tv ratio error: {e}")
        return None
