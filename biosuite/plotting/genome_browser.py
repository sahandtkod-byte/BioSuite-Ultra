"""
Genome browser track viewer — BAM coverage, VCF variants, BED intervals.
Pure Python implementation with matplotlib.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'parse_bed',
    'parse_vcf',
    'compute_coverage',
    'plot_genome_tracks',
    'create_coverage_from_bam',
    'create_bed_track',
    'create_variant_track',
]

def _note(problems, message):
    """Record why a row was not used, when the caller asked to be told."""
    if problems is not None:
        problems.append(message)


def parse_bed(path, max_regions=10000, problems=None):
    """Parse a BED file into a list of (chrom, start, end, name, score) tuples.

    Args:
        path: path to BED file.
        max_regions: maximum regions to read.
        problems: optional list that receives one line per row the parser could
            not use, and why.  A single unreadable row used to abort the whole
            file with ``int()``'s own message and no line number; a backwards
            interval used to be handed to the drawer as a negative-width box.
    """
    regions = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = line.strip()
            if not row or row.startswith('#') or row.startswith('track'):
                continue
            parts = row.split('\t')
            if len(parts) < 3:
                # A 'browser' directive or a malformed row: a BED record needs at
                # least chrom, start and end.
                _note(problems, f"line {lineno}: fewer than three columns")
                continue
            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3] if len(parts) > 3 else ""
                score = float(parts[4]) if len(parts) > 4 else 0.0
            except ValueError as exc:
                _note(problems, f"line {lineno}: {exc}")
                continue
            if start < 0 or end < start:
                _note(problems, f"line {lineno}: interval {start}-{end} does not "
                                "run forwards")
                continue
            regions.append((chrom, start, end, name, score))
            if len(regions) >= max_regions:
                _note(problems, f"stopped after {max_regions:,} regions")
                break
    return regions


def parse_vcf(path, max_variants=10000, problems=None):
    """Parse a VCF file into variant dicts.

    Returns list of dicts with keys: chrom, pos, ref, alt, qual, filter, info.

    Args:
        problems: optional list that receives why a row was not used (a
            non-numeric POS used to abort the file, and a truncated row used to
            vanish).
    """
    variants = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = line.strip()
            if not row or row.startswith('##'):
                continue
            if row.startswith('#'):
                continue
            parts = row.split('\t')
            if len(parts) < 5:
                _note(problems, f"line {lineno}: fewer than five columns")
                continue
            try:
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alt = parts[4]
                # CHROM, POS, ID, REF and ALT are the minimum a row needs to be
                # usable; QUAL, FILTER and INFO are optional in a truncated record.
                qual = float(parts[5]) if len(parts) > 5 and parts[5] != '.' else 0.0
            except ValueError as exc:
                _note(problems, f"line {lineno}: {exc}")
                continue
            if pos < 1:
                _note(problems, f"line {lineno}: POS {pos} is not a 1-based position")
                continue
            filt = parts[6] if len(parts) > 6 else ""
            info = parts[7] if len(parts) > 7 else ""
            variants.append({
                'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt,
                'qual': qual, 'filter': filt, 'info': info
            })
            if len(variants) >= max_variants:
                _note(problems, f"stopped after {max_variants:,} variants")
                break
    return variants


def _checked_region(region):
    """Refuse a region that is not (chrom, start, end), naming what arrived.

    ``region="chr1"`` used to be read as chrom 'c', start 'h', end 'r': every read
    was filtered out by a chromosome called 'c' and the plot came back empty.  An
    open region (``end`` of ``None``) is legal — the coverage path fills it in from
    the file's header — so only the start has to be a position.
    """
    if region is None:
        return None
    try:
        chrom, start, end = region
    except (TypeError, ValueError):
        raise ValueError(f"region must be (chrom, start, end), got {region!r}") from None
    if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
        raise ValueError(f"region start must be a whole base position, got {region!r}")
    if end is not None:
        if not isinstance(end, (int, np.integer)) or isinstance(end, bool):
            raise ValueError(f"region end must be a whole base position or None, "
                             f"got {region!r}")
        if end < start:
            raise ValueError(f"region {region!r} ends before it starts")
        end = int(end)
    return chrom, int(start), end


def compute_coverage(bam_path, region=None, bin_size=100):
    """Compute read coverage from a BAM/SAM file.

    Args:
        bam_path: path to BAM or SAM file.
        region: optional (chrom, start, end) tuple.
        bin_size: bin size in bp.

    Returns:
        positions, coverage arrays.

    Raises:
        ValueError: if *bin_size* is not a positive whole number, if *region* is
            not a (chrom, start, end) tuple, or if pysam is missing and the file
            is binary (a BAM read as text used to fail inside the UTF-8 decoder).
    """
    if isinstance(bin_size, bool) or not isinstance(bin_size, (int, np.integer)):
        raise ValueError(f"bin_size must be a whole number of bases, got {bin_size!r}")
    if bin_size < 1:
        raise ValueError(f"bin_size must be at least one base, got {bin_size}")
    checked = _checked_region(region)

    try:
        import pysam
        bam = pysam.AlignmentFile(bam_path, "rb")
    except ImportError:
        # Fallback: parse SAM as text
        return _compute_coverage_sam(bam_path, checked, bin_size)

    chroms = checked[0] if checked else None
    start = checked[1] if checked else 0
    end = checked[2] if checked else None

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
        # reference_end is exclusive in pysam: a read whose last base sits at 59
        # covers bin 5 (50-59), and counting 60 // bin_size handed every read that
        # ended on a bin boundary one phantom bin.
        ref_end = read.reference_end
        if ref_end is None:
            # No CIGAR: pysam cannot say where the read ends, so it cannot be binned.
            continue
        r_end = min(end - start, max(0, ref_end - 1 - start)) // bin_size
        for i in range(max(0, r_start), min(len(coverage), r_end + 1)):
            coverage[i] += 1

    positions = np.arange(len(coverage)) * bin_size + start
    return positions, coverage


def _compute_coverage_sam(path, region, bin_size):
    """Fallback SAM coverage computation, clipped to *region* when one is given.

    The chromosome name was the only part of a region this honoured: a window
    (chr1, 4900, 5100) returned bins from 0 and counted a read at position 100,
    while the pysam path clips to the same window.  A binary file (a real BAM)
    used to die inside the UTF-8 decoder with no mention of pysam.
    """
    coverage_dict = defaultdict(int)
    max_bin = -1
    region_start = region[1] if region else 0
    region_end = region[2] if region else None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith('@') or line.strip() == '':
                    continue
                parts = line.rstrip('\r\n').split('\t')
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
                clip_start = max(pos, region_start)
                clip_end = end if region_end is None else min(end, region_end)
                if clip_end <= clip_start:
                    continue

                # Bin-aligned counting: stepping by bin_size from the read
                # START skips bins when the read starts mid-bin (old code
                # undercounted every read not starting on a bin boundary).
                first = (clip_start - region_start) // bin_size
                last = (clip_end - 1 - region_start) // bin_size
                for b in range(first, last + 1):
                    coverage_dict[b] += 1
                max_bin = max(max_bin, last)
    except UnicodeDecodeError:
        raise ValueError(
            f"{path} is binary, so it is a BAM (or a compressed SAM), not the text "
            "SAM this fallback reads; install pysam (pip install pysam) or convert "
            "the file to SAM.") from None

    n_bins = max_bin + 1
    if region_end is not None:
        n_bins = max(n_bins, (region_end - region_start) // bin_size + 1)
    positions = np.arange(n_bins) * bin_size + region_start
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
        # N (spliced/skipped region) consumes reference exactly like D —
        # old code under-spanned spliced reads by the intron length.
        if op in ('M', 'D', 'N', '=', 'X'):
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
            - 'skipped': optional list of rows the loader could not use
        region: (chrom, start, end) or None.
        title: overall title.
        track_height: height per track in inches.

    Returns:
        The figure, or ``None`` when *tracks* is empty (the GUI checks for that).
    """
    region = _checked_region(region)
    n_tracks = len(tracks)
    if n_tracks == 0:
        return None

    fig, axes = plt.subplots(n_tracks, 1, figsize=(figsize[0], track_height * n_tracks),
                              sharex=True)
    if n_tracks == 1:
        axes = [axes]

    for i, (ax, track) in enumerate(zip(axes, tracks)):
        track_type = track.get('type', 'coverage')
        name = track.get('name', f'Track {i+1}')
        color = track.get('color', '#00ff88')
        data = track.get('data', {})
        if not isinstance(data, dict):
            raise ValueError(f"track {name!r} carries a {type(data).__name__} as 'data'; "
                             "a track's data is a dict of the arrays its type draws")

        note = None
        if track_type == 'coverage':
            note = _draw_coverage_track(ax, data, region, color)
        elif track_type == 'bed':
            note = _draw_bed_track(ax, data, region, color)
        elif track_type == 'variant':
            note = _draw_variant_track(ax, data, region, color)
        elif track_type == 'signal':
            note = _draw_signal_track(ax, data, region, color)
        else:
            # A row with nothing in it and no word is how a typo ('gene' for 'bed')
            # reads as "this region has no features".
            ax.text(0.5, 0.5, f"unknown track type {track_type!r}", ha='center',
                    va='center', transform=ax.transAxes, color='gray')

        skipped = track.get('skipped') or []
        if skipped:
            note = (f"{len(skipped)} row(s) skipped" if note is None
                    else f"{note}; {len(skipped)} row(s) skipped")

        label = name if not note else f"{name} — {note}"
        ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=80, va='center')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == n_tracks - 1:
            ax.set_xlabel("Position (bp)")
        else:
            ax.set_xticklabels([])

    if region is not None and region[2] is not None:
        # Every row shows the window that was asked for, whatever its type: a
        # BED-only view used to fall back to matplotlib's auto limits (0..1).
        axes[-1].set_xlim(region[1], region[2])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def _track_arrays(data, first, second, kind):
    """Read two arrays off a track and refuse them if they cannot be plotted."""
    a = data.get(first, np.array([]))
    b = data.get(second, np.array([]))
    if len(a) != len(b):
        raise ValueError(f"the {kind} track got {len(a)} {first} and {len(b)} "
                         f"{second}; every value needs a position")
    return a, b


def _draw_coverage_track(ax, data, region, color):
    """Draw a coverage/wiggle track."""
    positions, coverage = _track_arrays(data, 'positions', 'coverage', 'coverage')

    if len(positions) == 0 or len(coverage) == 0:
        ax.text(0.5, 0.5, "No coverage data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return None

    ax.fill_between(positions, 0, coverage, color=color, alpha=0.7)
    ax.plot(positions, coverage, color=color, linewidth=0.5)
    ax.set_ylim(0, max(coverage) * 1.1 if len(coverage) > 0 else 1)

    if region and region[2] is not None:
        ax.set_xlim(region[1], region[2])
    else:
        ax.set_xlim(positions[0], positions[-1] if len(positions) > 1 else 1)

    ax.set_ylabel("Coverage", fontsize=9)
    return None


def _draw_bed_track(ax, data, region, color):
    """Draw BED interval track, restricted to the region's chromosome when given."""
    regions = data.get('regions', [])
    if not regions:
        ax.text(0.5, 0.5, "No BED data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return None

    note = None
    if region:
        wanted = [r for r in regions if r[0] == region[0]]
        if len(wanted) != len(regions):
            note = f"{len(regions) - len(wanted)} region(s) on other chromosomes not shown"
        regions = wanted

    # Group by y-level to avoid overlap
    lanes = []
    sorted_regions = sorted(regions, key=lambda r: r[1])
    for r in sorted_regions:
        start, end = r[1], r[2]
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
    return note


def _variant_kind(ref, alt):
    """Classify a variant by the sizes of its alleles, commas included.

    A multi-allelic record (``A`` -> ``A,C``) is one SNP and one silent allele; the
    length comparison called it an insertion and coloured it green.
    """
    alleles = [a for a in alt.split(',') if a and a != '.']
    if not alleles:
        return 'SNP'
    if any(len(a) < len(ref) for a in alleles):
        return 'DEL'
    if any(len(a) > len(ref) for a in alleles):
        return 'INS'
    return 'SNP'


def _draw_variant_track(ax, data, region, color):
    """Draw variant markers, restricted to the region's chromosome when given."""
    variants = data.get('variants', [])
    if not variants:
        ax.text(0.5, 0.5, "No variant data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return None

    note = None
    if region and all('chrom' in v for v in variants):
        # A hand-built variant dict without a chromosome cannot be filtered by one,
        # so those records are all kept rather than all dropped.
        wanted = [v for v in variants if v['chrom'] == region[0]]
        if len(wanted) != len(variants):
            note = f"{len(variants) - len(wanted)} variant(s) on other chromosomes not shown"
        variants = wanted

    alt_colors = {'SNP': '#ff6b6b', 'DEL': '#ffd93d', 'INS': '#6bcb77'}
    for v in variants:
        pos = v['pos']
        ref = v.get('ref', '')
        alt = v.get('alt', '')
        vtype = _variant_kind(ref, alt)
        c = alt_colors.get(vtype, color)
        ax.plot(pos, 0.5, marker='^', color=c, markersize=8, markeredgecolor='black')
        ax.text(pos, 0.75, f"{ref}>{alt}", ha='center', va='bottom', fontsize=6,
                rotation=45)

    if region and region[2] is not None:
        ax.set_xlim(region[1], region[2])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_ylabel("Variants", fontsize=9)
    return note


def _draw_signal_track(ax, data, region, color):
    """Draw a generic signal track (e.g., ChIP-seq peaks)."""
    positions, values = _track_arrays(data, 'positions', 'values', 'signal')

    if len(positions) == 0:
        ax.text(0.5, 0.5, "No signal data", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return None

    ax.fill_between(positions, 0, values, color=color, alpha=0.6)
    ax.plot(positions, values, color=color, linewidth=0.8)
    ax.set_ylim(0, max(values) * 1.1 if len(values) > 0 else 1)

    if region and region[2] is not None:
        ax.set_xlim(region[1], region[2])
    ax.set_ylabel("Signal", fontsize=9)
    return None


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
    """Create BED track dict from BED file.

    The dict carries ``skipped``: one line per row the parser could not use, so the
    browser can say "3 row(s) skipped" instead of quietly drawing fewer features.
    """
    problems = []
    regions = parse_bed(bed_path, problems=problems)
    return {
        'type': 'bed',
        'name': 'BED',
        'data': {'regions': regions},
        'color': '#4ecdc4',
        'skipped': problems,
    }


def create_variant_track(vcf_path):
    """Create variant track dict from VCF file."""
    problems = []
    variants = parse_vcf(vcf_path, problems=problems)
    return {
        'type': 'variant',
        'name': 'Variants',
        'data': {'variants': variants},
        'color': '#ff6b6b',
        'skipped': problems,
    }
