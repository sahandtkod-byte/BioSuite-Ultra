"""
Genome browser track viewer — BAM coverage, VCF variants, BED intervals.
Pure Python implementation with matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict


def parse_bed(path, max_regions=10000):
    """Parse a BED file into a list of (chrom, start, end, name, score) tuples.

    Args:
        path: path to BED file.
        max_regions: maximum regions to read.
    """
    regions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track'):
                continue
            parts = line.split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else ""
            score = float(parts[4]) if len(parts) > 4 else 0.0
            regions.append((chrom, start, end, name, score))
            if len(regions) >= max_regions:
                break
    return regions


def parse_vcf(path, max_variants=10000):
    """Parse a VCF file into variant dicts.

    Returns list of dicts with keys: chrom, pos, ref, alt, qual, filter, info.
    """
    variants = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('##'):
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            qual = float(parts[5]) if parts[5] != '.' else 0.0
            filt = parts[6]
            info = parts[7] if len(parts) > 7 else ""
            variants.append({
                'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt,
                'qual': qual, 'filter': filt, 'info': info
            })
            if len(variants) >= max_variants:
                break
    return variants


def compute_coverage(bam_path, region=None, bin_size=100):
    """Compute read coverage from a BAM/SAM file.

    Args:
        bam_path: path to BAM or SAM file.
        region: optional (chrom, start, end) tuple.
        bin_size: bin size in bp.

    Returns:
        positions, coverage arrays.
    """
    try:
        import pysam
        bam = pysam.AlignmentFile(bam_path, "rb")
    except ImportError:
        # Fallback: parse SAM as text
        return _compute_coverage_sam(bam_path, region, bin_size)

    chroms = region[0] if region else None
    start = region[1] if region else 0
    end = region[2] if region else None

    if end is None and chroms:
        end = bam.get_reference_length(chroms)
    elif end is None:
        # Get max length from header
        for ref in bam.references:
            ref_len = bam.get_reference_length(ref)
            if end is None or ref_len > end:
                end = ref_len

    if end is None or end == 0:
        return np.array([]), np.array([])

    coverage = np.zeros(max(1, end - start) // bin_size + 1)

    for read in bam.fetch(chroms, start, end) if chroms else bam.fetch():
        r_start = max(0, read.reference_start - start) // bin_size
        r_end = min(end - start, read.reference_end - start) // bin_size
        for i in range(max(0, r_start), min(len(coverage), r_end + 1)):
            coverage[i] += 1

    positions = np.arange(len(coverage)) * bin_size + start
    return positions, coverage


def _compute_coverage_sam(path, region, bin_size):
    """Fallback SAM coverage computation."""
    coverage_dict = defaultdict(int)
    max_end = 0
    with open(path) as f:
        for line in f:
            if line.startswith('@') or line.strip() == '':
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            flag = int(parts[1])
            if flag & 4:  # unmapped
                continue
            chrom = parts[2]
            pos = int(parts[3]) - 1  # 0-based
            cigar = parts[5]
            read_len = _parse_cigar_length(cigar)
            if read_len == 0:
                read_len = len(parts[9]) if len(parts) > 9 else 100

            if region and chrom != region[0]:
                continue

            end = pos + read_len
            if end > max_end:
                max_end = end

            for bp in range(pos, end, bin_size):
                coverage_dict[bp // bin_size] += 1

    n_bins = max_end // bin_size + 1
    positions = np.arange(n_bins) * bin_size
    coverage = np.array([coverage_dict.get(i, 0) for i in range(n_bins)])
    return positions, coverage


def _parse_cigar_length(cigar):
    """Get reference length from CIGAR string."""
    import re
    nums = re.findall(r'\d+', cigar)
    ops = re.findall(r'[MIDNSHP=X]', cigar)
    length = 0
    for n, op in zip(nums, ops):
        n = int(n)
        if op in ('M', 'D', '=', 'X'):
            length += n
    return length


def plot_genome_tracks(tracks, region=None, title="Genome Browser",
                       figsize=(14, 8), track_height=1.5):
    """Plot a genome browser view with multiple tracks.

    Args:
        tracks: list of track dicts, each with:
            - 'type': 'coverage', 'bed', 'variant', 'signal'
            - 'name': track label
            - 'data': track-specific data
            - 'color': optional color
        region: (chrom, start, end) or None.
        title: overall title.
        track_height: height per track in inches.
    """
    n_tracks = len(tracks)
    if n_tracks == 0:
        return

    fig, axes = plt.subplots(n_tracks, 1, figsize=(figsize[0], track_height * n_tracks),
                              sharex=True)
    if n_tracks == 1:
        axes = [axes]

    for i, (ax, track) in enumerate(zip(axes, tracks)):
        track_type = track.get('type', 'coverage')
        name = track.get('name', f'Track {i+1}')
        color = track.get('color', '#00ff88')
        data = track.get('data', {})

        ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=80, va='center')

        if track_type == 'coverage':
            _draw_coverage_track(ax, data, region, color)
        elif track_type == 'bed':
            _draw_bed_track(ax, data, region, color)
        elif track_type == 'variant':
            _draw_variant_track(ax, data, region, color)
        elif track_type == 'signal':
            _draw_signal_track(ax, data, region, color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == n_tracks - 1:
            ax.set_xlabel("Position (bp)")
        else:
            ax.set_xticklabels([])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def _draw_coverage_track(ax, data, region, color):
    """Draw a coverage/wiggle track."""
    positions = data.get('positions', np.array([]))
    coverage = data.get('coverage', np.array([]))

    if len(positions) == 0 or len(coverage) == 0:
        ax.text(0.5, 0.5, "No coverage data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return

    ax.fill_between(positions, 0, coverage, color=color, alpha=0.7)
    ax.plot(positions, coverage, color=color, linewidth=0.5)
    ax.set_ylim(0, max(coverage) * 1.1 if len(coverage) > 0 else 1)

    if region:
        ax.set_xlim(region[1], region[2])
    else:
        ax.set_xlim(positions[0], positions[-1] if len(positions) > 1 else 1)

    ax.set_ylabel("Coverage", fontsize=9)


def _draw_bed_track(ax, data, region, color):
    """Draw BED interval track."""
    regions = data.get('regions', [])
    if not regions:
        ax.text(0.5, 0.5, "No BED data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return

    # Group by y-level to avoid overlap
    lanes = []
    sorted_regions = sorted(regions, key=lambda r: r[1])
    for r in sorted_regions:
        chrom, start, end = r[0], r[1], r[2]
        name = r[3] if len(r) > 3 else ""
        placed = False
        for lane_idx, lane_end in enumerate(lanes):
            if start >= lane_end:
                lanes[lane_idx] = end
                y = lane_idx
                placed = True
                break
        if not placed:
            y = len(lanes)
            lanes.append(end)

        rect = Rectangle((start, y), end - start, 0.6,
                          facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        if end - start > 0:
            ax.text((start + end) / 2, y + 0.3, name[:15], ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    ax.set_ylim(-0.5, max(len(lanes), 1))
    ax.set_yticks([])
    ax.set_ylabel("BED", fontsize=9)


def _draw_variant_track(ax, data, region, color):
    """Draw variant markers."""
    variants = data.get('variants', [])
    if not variants:
        ax.text(0.5, 0.5, "No variant data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return

    alt_colors = {'SNP': '#ff6b6b', 'DEL': '#ffd93d', 'INS': '#6bcb77'}
    for v in variants:
        pos = v['pos']
        ref = v.get('ref', '')
        alt = v.get('alt', '')
        vtype = 'DEL' if len(alt) < len(ref) else ('INS' if len(alt) > len(ref) else 'SNP')
        c = alt_colors.get(vtype, color)
        ax.plot(pos, 0.5, marker='^', color=c, markersize=8, markeredgecolor='black')
        ax.text(pos, 0.75, f"{ref}>{alt}", ha='center', va='bottom', fontsize=6,
                rotation=45)

    if region:
        ax.set_xlim(region[1], region[2])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_ylabel("Variants", fontsize=9)


def _draw_signal_track(ax, data, region, color):
    """Draw a generic signal track (e.g., ChIP-seq peaks)."""
    positions = data.get('positions', np.array([]))
    values = data.get('values', np.array([]))

    if len(positions) == 0:
        ax.text(0.5, 0.5, "No signal data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return

    ax.fill_between(positions, 0, values, color=color, alpha=0.6)
    ax.plot(positions, values, color=color, linewidth=0.8)
    ax.set_ylim(0, max(values) * 1.1 if len(values) > 0 else 1)

    if region:
        ax.set_xlim(region[1], region[2])
    ax.set_ylabel("Signal", fontsize=9)


def create_coverage_from_bam(bam_path, region=None, bin_size=100):
    """Create coverage track dict from BAM file."""
    positions, coverage = compute_coverage(bam_path, region, bin_size)
    return {
        'type': 'coverage',
        'name': 'Coverage',
        'data': {'positions': positions, 'coverage': coverage},
        'color': '#00ff88'
    }


def create_bed_track(bed_path):
    """Create BED track dict from BED file."""
    regions = parse_bed(bed_path)
    return {
        'type': 'bed',
        'name': 'BED',
        'data': {'regions': regions},
        'color': '#4ecdc4'
    }


def create_variant_track(vcf_path):
    """Create variant track dict from VCF file."""
    variants = parse_vcf(vcf_path)
    return {
        'type': 'variant',
        'name': 'Variants',
        'data': {'variants': variants},
        'color': '#ff6b6b'
    }
