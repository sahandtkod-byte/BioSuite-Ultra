"""
ORF (Open Reading Frame) finder, restriction enzyme mapper, and primer design.

Pure Python implementations for common sequence analysis tasks.
"""
import re
from dataclasses import dataclass


@dataclass
class ORF:
    frame: int
    start: int
    end: int
    length: int
    protein: str
    has_start_codon: bool
    has_stop_codon: bool


@dataclass
class RestrictionSite:
    enzyme: str
    sequence: str
    cut_position: int
    position: int
    strand: str


@dataclass
class Primer:
    name: str
    sequence: str
    length: int
    gc_content: float
    tm: float
    position: int
    strand: str


from .utils import GENETIC_CODE, RESTRICTION_ENZYMES, reverse_complement_dna


def find_orfs(sequence, min_length=30, include_start=False):
    """Find all open reading frames in a sequence.

    Args:
        sequence: DNA sequence string.
        min_length: Minimum protein length in amino acids.
        include_start: If True, only ORFs starting with ATG.  If False,
            report every stop-to-stop frame segment (``has_start_codon``
            then reports whether the segment happens to begin with M).
            NOTE: the old implementation IGNORED this parameter — ORFs
            always required ATG regardless of the flag; pass
            ``include_start=True`` to reproduce that behaviour.
        Only the longest ORF per stop-to-stop segment is reported
        (nested ATGs are not expanded).

    Returns:
        List of ORF objects, longest first.
    """
    seq = sequence.upper().replace('U', 'T')
    orfs = []

    def _segment_orfs(frame_start, seg_start, seg_end, has_stop, frame):
        """Translate seq[seg_start:seg_end] and emit ORFs if long enough.

        Emits the full stop-to-stop segment plus (when different) the
        longest nested ORF starting at an in-frame ATG — mirroring NCBI
        ORFfinder's 'nested ORF' view.
        """
        protein = ''.join(
            GENETIC_CODE.get(seq[j:j+3], 'X')
            for j in range(seg_start, seg_end, 3)
            if j + 3 <= len(seq)
        )
        while protein.startswith('X'):
            seg_start += 3
            protein = protein[1:]

        if include_start is True or include_start == 'ATG':
            m_pos = protein.find('M')
            if m_pos < 0:
                return
            variants = {(m_pos, protein[m_pos:])}
        else:
            variants = {(0, protein)}
            m_pos = protein.find('M')
            if m_pos > 0:  # nested-ATG view (m_pos==0 is the segment itself)
                variants.add((m_pos, protein[m_pos:]))

        for offset, prot in sorted(variants):
            if len(prot) < min_length or not prot:
                continue
            orfs.append(ORF(
                frame=frame + 1,
                start=seg_start + offset * 3,
                end=seg_end + (3 if has_stop else 0),
                length=len(prot),
                protein=prot,
                has_start_codon=prot.startswith('M'),
                has_stop_codon=has_stop
            ))

    for frame in range(3):
        seg_start = frame
        j = frame
        while j <= len(seq) - 3:
            aa = GENETIC_CODE.get(seq[j:j+3], 'X')
            if aa == '*':
                _segment_orfs(frame, seg_start, j, True, frame)
                seg_start = j + 3
            j += 3
        _segment_orfs(frame, seg_start, j, False, frame)

    orfs.sort(key=lambda o: o.length, reverse=True)
    return orfs


def find_restriction_sites(sequence, enzymes=None):
    """Find restriction enzyme cut sites in a sequence.

    Args:
        sequence: DNA sequence string.
        enzymes: List of enzyme names (default: all known enzymes).

    Returns:
        List of RestrictionSite objects.
    """
    seq = sequence.upper()
    if enzymes is None:
        enzymes = list(RESTRICTION_ENZYMES.keys())

    sites = []
    for enzyme_name in enzymes:
        if enzyme_name not in RESTRICTION_ENZYMES:
            continue
        pattern, cut_pos = RESTRICTION_ENZYMES[enzyme_name]
        for match in re.finditer(f'(?=({pattern}))', seq):
            sites.append(RestrictionSite(
                enzyme=enzyme_name,
                sequence=pattern,
                cut_position=cut_pos,
                position=match.start(),
                strand='+'
            ))
        # Reverse-strand sites: scan rc ONCE per enzyme.  This block used to
        # sit inside the '+' match loop, so every '+' hit re-added ALL '-'
        # sites (non-palindromic enzymes were massively over-counted).
        rc = _reverse_complement(pattern)
        if rc != pattern:
            for m2 in re.finditer(f'(?=({rc}))', seq):
                sites.append(RestrictionSite(
                    enzyme=enzyme_name,
                    sequence=rc,
                    cut_position=len(pattern) - cut_pos,
                    position=m2.start(),
                    strand='-'
                ))

    sites.sort(key=lambda s: s.position)
    return sites


def design_primers(target_seq, amplicon_start=0, amplicon_end=None,
                   primer_length=20, min_tm=55, max_tm=65, gc_range=(40, 70)):
    """Design PCR primers for a target region.

    Args:
        target_seq: Template DNA sequence.
        amplicon_start: Start position of amplicon.
        amplicon_end: End position of amplicon (default: end of sequence).
        primer_length: Desired primer length.
        min_tm: Minimum melting temperature.
        max_tm: Maximum melting temperature.
        gc_range: Tuple of (min, max) GC percentage.

    Returns:
        Tuple of (forward_primer, reverse_primer) Primer objects.
    """
    seq = target_seq.upper()
    if amplicon_end is None:
        amplicon_end = len(seq)

    fwd = _find_primer(seq, amplicon_start, amplicon_end, primer_length,
                       min_tm, max_tm, gc_range, strand='+')
    rev = _find_primer(seq, amplicon_start, amplicon_end, primer_length,
                       min_tm, max_tm, gc_range, strand='-')

    return fwd, rev


def _find_primer(seq, start, end, length, min_tm, max_tm, gc_range, strand):
    """Search for a valid primer in the given region.

    Forward primers anchor near the amplicon START, reverse primers near
    the amplicon END — the old scoring pulled BOTH towards the middle of
    the region, producing colliding/overlapping primer pairs.
    """
    best = None
    best_score = -1
    target_pos = start if strand == '+' else max(start, end - length)

    for i in range(start, min(end - length + 1, len(seq) - length + 1)):
        if strand == '+':
            primer_seq = seq[i:i + length]
        else:
            primer_seq = _reverse_complement(seq[i:i + length])

        gc = (primer_seq.count('G') + primer_seq.count('C')) / len(primer_seq) * 100
        tm = _calculate_tm(primer_seq)

        if gc < gc_range[0] or gc > gc_range[1]:
            continue
        if tm < min_tm or tm > max_tm:
            continue

        # Score: prefer the correct flank, stable GC, good Tm
        pos_score = 1.0 - abs(i - target_pos) / max((end - start) / 2, 1)
        gc_score = 1.0 - abs(gc - 55) / 45
        tm_score = 1.0 - abs(tm - 60) / 10
        score = pos_score + gc_score + tm_score

        if score > best_score:
            best_score = score
            best = Primer(
                name=f"{'FWD' if strand == '+' else 'REV'}_{i}",
                sequence=primer_seq,
                length=len(primer_seq),
                gc_content=round(gc, 1),
                tm=round(tm, 1),
                position=i,
                strand=strand
            )

    return best


def _calculate_tm(primer_seq):
    """Calculate melting temperature using Wallace rule (short primers)."""
    gc = primer_seq.count('G') + primer_seq.count('C')
    at = len(primer_seq) - gc
    if len(primer_seq) < 14:
        return 2 * at + 4 * gc
    else:
        return 64.9 + 41 * (gc - 16.4) / len(primer_seq)


# Use shared reverse_complement_dna from utils
_reverse_complement = reverse_complement_dna


def format_orf_results(orfs):
    lines = ["=== Open Reading Frames ===", f"Found {len(orfs)} ORFs\n"]
    lines.append(f"{'#':<4} {'Frame':<7} {'Start':>7} {'End':>7} {'Length':>8} {'Start?':>7} {'Stop?':>6}")
    lines.append("-" * 55)
    for i, orf in enumerate(orfs[:20]):
        lines.append(f"{i+1:<4} {orf.frame:<7} {orf.start:>7} {orf.end:>7} {orf.length:>8} "
                    f"{'Yes' if orf.has_start_codon else 'No':>7} {'Yes' if orf.has_stop_codon else 'No':>6}")
    return '\n'.join(lines)


def format_restriction_sites(sites):
    lines = ["=== Restriction Sites ===", f"Found {len(sites)} cut sites\n"]
    lines.append(f"{'Enzyme':<12} {'Sequence':<12} {'Position':>10} {'Strand':>7}")
    lines.append("-" * 45)
    for s in sites[:30]:
        lines.append(f"{s.enzyme:<12} {s.sequence:<12} {s.position:>10} {s.strand:>7}")
    return '\n'.join(lines)


def format_primers(fwd, rev):
    lines = ["=== Designed Primers ==="]
    if fwd:
        lines.append(f"\nForward: {fwd.sequence}")
        lines.append(f"  Length: {fwd.length} bp, GC: {fwd.gc_content}%, Tm: {fwd.tm}°C")
        lines.append(f"  Position: {fwd.position}")
    if rev:
        lines.append(f"\nReverse: {rev.sequence}")
        lines.append(f"  Length: {rev.length} bp, GC: {rev.gc_content}%, Tm: {rev.tm}°C")
        lines.append(f"  Position: {rev.position}")
    if fwd and rev:
        lines.append(f"\nAmplicon size: ~{rev.position - fwd.position + rev.length} bp")
    return '\n'.join(lines)
