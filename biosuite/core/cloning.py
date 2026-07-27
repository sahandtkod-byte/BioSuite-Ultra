"""
Molecular Cloning Toolkit — restriction digestion, ligation, PCR, and
virtual gel electrophoresis.

Pure matplotlib / numpy. No paid dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Restriction enzyme database (imported from shared utils)
# ---------------------------------------------------------------------------
# We need BOTH the sites and the cut positions for accurate digestion.
from biosuite.core.utils import RESTRICTION_ENZYMES


# ---------------------------------------------------------------------------
# DNA validation helper
# ---------------------------------------------------------------------------
_VALID_DNA = set("ATCGWSMKRYBDHVN")

def _validate_dna(seq: str, name: str = "sequence") -> str:
    """Validate and clean a DNA sequence. Raises ValueError on bad input."""
    seq = seq.upper().strip()
    if not seq:
        raise ValueError(f"{name} is empty")
    bad = set(seq) - _VALID_DNA
    if bad:
        raise ValueError(
            f"Invalid characters in {name}: {''.join(sorted(bad))}. "
            "Only IUPAC nucleotide codes are accepted."
        )
    return seq


# ---------------------------------------------------------------------------
# Reverse complement helper
# ---------------------------------------------------------------------------
_COMPLEMENT = str.maketrans("ATCGWSMKRYBDHVN",
                             "TAGCWSMKRYBDHVN")

def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.upper().translate(_COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# Restriction site search
# ---------------------------------------------------------------------------
def find_restriction_sites(sequence: str,
                           enzyme: Optional[str] = None) -> Union[List[int], Dict[str, List[int]]]:
    """
    Find all restriction sites in *sequence*.

    Parameters
    ----------
    sequence : str
        DNA sequence (will be upper-cased internally).
    enzyme : str, optional
        Single enzyme name to search for. If None, searches all enzymes.

    Returns
    -------
    list of int  (when enzyme is specified)
        0-based positions where the recognition site starts.
    dict of {enzyme_name: list of int}  (when enzyme is None)
        Maps each enzyme to its hit positions.

    Raises
    ------
    ValueError
        If *enzyme* is not in the restriction enzyme database.
    """
    seq = _validate_dna(sequence, "search sequence")

    if enzyme is not None:
        if enzyme not in RESTRICTION_ENZYMES:
            raise ValueError(
                f"Unknown enzyme: {enzyme}. "
                f"Available: {', '.join(sorted(RESTRICTION_ENZYMES.keys())[:10])}..."
            )
        site = RESTRICTION_ENZYMES[enzyme][0].upper()
        return _find_all(seq, site)

    # Return per-enzyme results
    results: Dict[str, List[int]] = {}
    for name, (site, _cut) in RESTRICTION_ENZYMES.items():
        hits = _find_all(seq, site.upper())
        if hits:
            results[name] = hits
    return results


def _find_all(seq: str, pattern: str) -> List[int]:
    """Find all start positions of *pattern* in *seq* (overlapping)."""
    positions: List[int] = []
    start = 0
    while True:
        pos = seq.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


# ---------------------------------------------------------------------------
# Restriction digestion
# ---------------------------------------------------------------------------
def simulate_digestion(sequence: str,
                       enzyme: Union[str, List[str]],
                       topology: str = "circular") -> dict:
    """
    Simulate restriction enzyme digestion.

    Parameters
    ----------
    sequence : str
        DNA sequence to digest.
    enzyme : str or list of str
        Enzyme name(s) (must exist in RESTRICTION_ENZYMES).
    topology : str
        "circular" or "linear".

    Returns
    -------
    dict with keys:
        fragments : list of dict  — each dict has 'size', 'sequence', 'start', 'end'
        sizes : list of int       — fragment sizes in bp (for backward compat)
        total_bp : int            — total sequence length
        cuts : list of dict       — cut info dicts {enzyme, position, cut_offset}
        enzymes_used : list       — enzyme names used
        overhangs : dict          — enzyme_name -> overhang type ('5prime', '3prime', 'blunt')
    """
    seq = _validate_dna(sequence, "digestion template")
    total = len(seq)

    enzyme_list = [enzyme] if isinstance(enzyme, str) else list(enzyme)

    # Resolve enzyme names to (site, cut_position) tuples
    enzyme_info: Dict[str, Tuple[str, int]] = {}
    for enz in enzyme_list:
        if enz not in RESTRICTION_ENZYMES:
            raise ValueError(
                f"Unknown enzyme: {enz}. "
                f"Available: {', '.join(sorted(RESTRICTION_ENZYMES.keys())[:10])}..."
            )
        enzyme_info[enz] = RESTRICTION_ENZYMES[enz]

    # Find cut positions with enzyme attribution
    # cut_position in RESTRICTION_ENZYMES means: cut happens after
    # this many bases from the start of the recognition site.
    cut_data: List[Dict] = []  # [{enzyme, site_start, cut_offset, cut_pos}]
    for enz_name, (site, cut_offset) in enzyme_info.items():
        site_upper = site.upper()
        for pos in _find_all(seq, site_upper):
            cut_data.append({
                "enzyme": enz_name,
                "site_start": pos,
                "cut_offset": cut_offset,
                "cut_pos": pos + cut_offset,
            })

    # Sort by cut position and deduplicate
    cut_data.sort(key=lambda c: c["cut_pos"])
    seen_positions = set()
    unique_cuts = []
    for c in cut_data:
        if c["cut_pos"] not in seen_positions:
            unique_cuts.append(c)
            seen_positions.add(c["cut_pos"])
    cut_data = unique_cuts

    # Classify overhang types
    overhangs: Dict[str, str] = {}
    for enz_name, (site, cut_offset) in enzyme_info.items():
        site_len = len(site)
        if cut_offset == 0 or cut_offset >= site_len:
            overhangs[enz_name] = "blunt"
        elif cut_offset <= site_len // 2:
            overhangs[enz_name] = "5prime"
        else:
            overhangs[enz_name] = "3prime"

    if not cut_data:
        return {
            "fragments": [{"size": total, "sequence": seq,
                           "start": 0, "end": total}],
            "sizes": [total],
            "total_bp": total,
            "cuts": [],
            "enzymes_used": enzyme_list,
            "overhangs": overhangs,
        }

    cut_positions = [c["cut_pos"] for c in cut_data]

    # Generate fragments
    if topology == "circular":
        fragments = []
        for i in range(len(cut_positions)):
            start = cut_positions[i]
            end = cut_positions[(i + 1) % len(cut_positions)]
            if end > start:
                frag_seq = seq[start:end]
                fragments.append({
                    "size": end - start,
                    "sequence": frag_seq,
                    "start": start,
                    "end": end,
                })
            else:
                # Wraps around
                frag_seq = seq[start:] + seq[:end]
                fragments.append({
                    "size": (total - start) + end,
                    "sequence": frag_seq,
                    "start": start,
                    "end": end,
                })
    else:
        # Linear: pieces between cuts + ends
        fragments = []
        positions = [0] + cut_positions + [total]
        for i in range(len(positions) - 1):
            s, e = positions[i], positions[i + 1]
            if e > s:
                fragments.append({
                    "size": e - s,
                    "sequence": seq[s:e],
                    "start": s,
                    "end": e,
                })

    sizes = [f["size"] for f in fragments]

    return {
        "fragments": fragments,
        "sizes": sizes,
        "total_bp": total,
        "cuts": cut_data,
        "enzymes_used": enzyme_list,
        "overhangs": overhangs,
    }


# ---------------------------------------------------------------------------
# Ligation
# ---------------------------------------------------------------------------
def simulate_ligation(fragments: List[str],
                      circular: bool = True,
                      vector_sequence: Optional[str] = None,
                      insert_sequence: Optional[str] = None) -> dict:
    """
    Simulate ligation of DNA fragments with compatibility checking.

    Parameters
    ----------
    fragments : list of str
        DNA fragment sequences to ligate.
    circular : bool
        Whether the final product is circular.
    vector_sequence : str, optional
        Vector backbone for insert verification.
    insert_sequence : str, optional
        Insert to verify was ligated correctly.

    Returns
    -------
    dict with keys:
        product : str          — ligated sequence
        size : int             — total product size
        fragment_count : int   — number of fragments ligated
        is_circular : bool     — whether product is circular
        efficiency : float     — estimated ligation efficiency (0-1)
        verified : bool        — True if insert found in product (when applicable)
        warnings : list of str — any warnings about the ligation
    """
    warnings = []

    if not fragments:
        return {
            "product": "",
            "size": 0,
            "fragment_count": 0,
            "is_circular": circular,
            "efficiency": 0.0,
            "verified": False,
            "warnings": ["No fragments provided for ligation."],
        }

    # Validate all fragments
    clean_frags = []
    for i, frag in enumerate(fragments):
        try:
            clean = _validate_dna(frag, f"fragment {i + 1}")
            clean_frags.append(clean)
        except ValueError as e:
            warnings.append(f"Fragment {i + 1} skipped: {e}")

    if not clean_frags:
        return {
            "product": "",
            "size": 0,
            "fragment_count": 0,
            "is_circular": circular,
            "efficiency": 0.0,
            "verified": False,
            "warnings": warnings + ["All fragments were invalid."],
        }

    # Estimate ligation efficiency
    # Efficiency depends on:
    # - Number of fragments (more = lower efficiency)
    # - Presence of compatible ends (estimated from fragment termini)
    # - Circular vs linear (circular requires all ends to match)
    n_frags = len(clean_frags)
    base_efficiency = 0.95
    frag_penalty = max(0, (n_frags - 1) * 0.08)
    circular_penalty = 0.15 if circular and n_frags > 1 else 0.0

    # Check end compatibility between adjacent fragments
    compat_bonus = 0.0
    for i in range(n_frags - 1):
        end_3 = clean_frags[i][-4:] if len(clean_frags[i]) >= 4 else clean_frags[i]
        start_5 = clean_frags[i + 1][:4] if len(clean_frags[i + 1]) >= 4 else clean_frags[i + 1]
        # Simple heuristic: partial complementary at junction improves efficiency
        rc_end = _reverse_complement(end_3)
        if start_5[:2] in rc_end or rc_end[:2] in start_5:
            compat_bonus += 0.03

    efficiency = max(0.05, min(1.0, base_efficiency - frag_penalty
                               - circular_penalty + compat_bonus))

    # Concatenate fragments
    product = "".join(clean_frags)

    # Verify insert if provided
    verified = False
    if insert_sequence and vector_sequence:
        try:
            ins = _validate_dna(insert_sequence, "insert")
            vec = _validate_dna(vector_sequence, "vector")
            # Check if the insert (or its RC) is present in the product
            verified = (ins in product) or (_reverse_complement(ins) in product)
            if not verified:
                # Also check if it appears split across junction
                half = len(ins) // 2
                if half > 4:
                    front = ins[:half]
                    back = ins[half:]
                    if (front in product and back in product):
                        verified = True
                        warnings.append(
                            "Insert found but may span a junction — verify orientation."
                        )
                if not verified:
                    warnings.append(
                        "WARNING: Insert NOT detected in ligated product! "
                        "Ligation may have failed or insert was lost."
                    )
        except ValueError as e:
            warnings.append(f"Insert verification skipped: {e}")

    return {
        "product": product,
        "size": len(product),
        "fragment_count": n_frags,
        "is_circular": circular,
        "efficiency": round(efficiency, 3),
        "verified": verified,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Primer design
# ---------------------------------------------------------------------------
def _gc_content(seq: str) -> float:
    s = seq.upper()
    gc = sum(1 for c in s if c in "GC")
    total = len([c for c in s if c in "ATCG"])
    return gc / total if total else 0.0


def _melting_temp(seq: str) -> float:
    """Calculate melting temperature using the nearest-neighbor approximation
    (simplified) for long primers and Wallace rule for short ones."""
    s = seq.upper()
    nA = s.count("A"); nT = s.count("T")
    nG = s.count("G"); nC = s.count("C")
    length = nA + nT + nG + nC
    if length == 0:
        return 0.0
    if length < 14:
        # Wallace rule (approximate)
        return 2.0 * (nA + nT) + 4.0 * (nG + nC)
    # SantaLucia 1998 unified parameters (simplified)
    return 64.9 + 41.0 * (nG + nC - 16.4) / length


def design_primers(target_seq: str,
                   primer_length: int = 20) -> dict:
    """
    Design forward and reverse primers from the start and end of *target_seq*.

    Returns
    -------
    dict with keys:
        forward : str  — forward primer sequence
        reverse : str  — reverse primer sequence (reverse complement)
        fwd_tm  : float
        rev_tm  : float
        fwd_gc  : float
        rev_gc  : float
        warnings : list of str
    """
    seq = _validate_dna(target_seq, "primer template")
    warnings = []

    if len(seq) < 2 * primer_length:
        warnings.append(
            f"Target sequence ({len(seq)} bp) shorter than 2× primer length "
            f"({2 * primer_length} bp). Primers may overlap."
        )

    if len(seq) < primer_length:
        raise ValueError(
            f"Target sequence ({len(seq)} bp) is shorter than primer length "
            f"({primer_length} bp)."
        )

    fwd_seq = seq[:primer_length]
    # Reverse primer: reverse complement of the last `primer_length` bases
    end_region = seq[-primer_length:]
    rev_seq = _reverse_complement(end_region)

    fwd_gc = _gc_content(fwd_seq)
    rev_gc = _gc_content(rev_seq)
    fwd_tm = _melting_temp(fwd_seq)
    rev_tm = _melting_temp(rev_seq)

    if fwd_gc < 0.4 or fwd_gc > 0.6:
        warnings.append(f"Forward primer GC content ({fwd_gc * 100:.1f}%) outside ideal range (40-60%).")
    if rev_gc < 0.4 or rev_gc > 0.6:
        warnings.append(f"Reverse primer GC content ({rev_gc * 100:.1f}%) outside ideal range (40-60%).")
    if abs(fwd_tm - rev_tm) > 5:
        warnings.append(
            f"Tm difference ({abs(fwd_tm - rev_tm):.1f}°C) exceeds recommended 5°C threshold."
        )

    return {
        "forward": fwd_seq,
        "reverse": rev_seq,
        "fwd_tm": round(fwd_tm, 1),
        "rev_tm": round(rev_tm, 1),
        "fwd_gc": round(fwd_gc * 100, 1),
        "rev_gc": round(rev_gc * 100, 1),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# PCR simulation
# ---------------------------------------------------------------------------
def simulate_pcr(template: str,
                 fwd_primer: str,
                 rev_primer: str,
                 cycles: int = 30) -> dict:
    """
    Simulate PCR amplification.

    Finds primer binding sites and returns the amplified product.
    Validates that forward primer binds upstream of reverse primer.

    Returns
    -------
    dict with keys:
        product : str      — amplified DNA sequence
        size : int         — product length in bp
        cycles : int       — number of cycles used
        fwd_pos : int      — forward primer binding position
        rev_pos : int      — reverse primer binding position (on template)
        efficiency : float — estimated per-cycle amplification efficiency
        warnings : list of str
    """
    seq = _validate_dna(template, "PCR template")
    fwd = _validate_dna(fwd_primer, "forward primer")
    rev = _validate_dna(rev_primer, "reverse primer")
    warnings = []

    if cycles < 1 or cycles > 50:
        warnings.append(f"Unusual cycle count: {cycles}. Typical range is 25-35.")

    fwd_pos = seq.find(fwd)
    if fwd_pos == -1:
        # Try with some mismatch tolerance (last 6 bases are most critical)
        if len(fwd) > 6:
            fwd_suffix = fwd[-6:]
            alt_pos = seq.find(fwd_suffix)
            if alt_pos != -1:
                warnings.append(
                    f"Forward primer not found exactly. Closest partial match "
                    f"at position {alt_pos - len(fwd) + 6}."
                )
        raise ValueError(
            f"Forward primer '{fwd}' not found in template. "
            "Check primer sequence and template."
        )

    # Reverse primer binds to the opposite strand → find its RC on the template
    rev_rc = _reverse_complement(rev)
    rev_pos = seq.find(rev_rc)
    if rev_pos == -1:
        if len(rev_rc) > 6:
            rev_suffix = rev_rc[-6:]
            alt_pos = seq.find(rev_suffix)
            if alt_pos != -1:
                warnings.append(
                    f"Reverse primer binding site not found exactly. "
                    f"Closest partial match at position {alt_pos - len(rev_rc) + 6}."
                )
        raise ValueError(
            f"Reverse primer binding site (RC: '{rev_rc}') not found in template. "
            "Check primer sequence and template."
        )

    # Validate primer orientation: forward must bind before reverse
    if fwd_pos >= rev_pos:
        raise ValueError(
            f"Primer orientation error: forward primer binds at position {fwd_pos} "
            f"but reverse primer RC binds at position {rev_pos}. "
            "Forward primer must bind upstream of the reverse primer binding site."
        )

    product_start = fwd_pos
    product_end = rev_pos + len(rev_rc)

    amplicon = seq[product_start:product_end]

    # Estimate amplification efficiency (based on product length)
    # Longer products amplify less efficiently
    product_len = len(amplicon)
    if product_len < 500:
        efficiency = 0.95
    elif product_len < 2000:
        efficiency = 0.90
    elif product_len < 5000:
        efficiency = 0.80
    else:
        efficiency = 0.65

    if product_len > 10000:
        warnings.append(
            f"Product is {product_len:,} bp. Very long amplicons may "
            "require optimization (touchdown PCR, DMSO, etc.)."
        )

    return {
        "product": amplicon,
        "size": len(amplicon),
        "cycles": cycles,
        "fwd_pos": fwd_pos,
        "rev_pos": rev_pos,
        "efficiency": efficiency,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Insert verification
# ---------------------------------------------------------------------------
def verify_insert(product: str,
                  expected_insert: str,
                  vector_backbone: Optional[str] = None) -> dict:
    """
    Verify that an insert is present and correctly oriented in a ligated product.

    Parameters
    ----------
    product : str
        The ligated product sequence.
    expected_insert : str
        The expected insert sequence.
    vector_backbone : str, optional
        If provided, checks that the insert is between vector segments.

    Returns
    -------
    dict with keys:
        insert_found : bool
        forward_orientation : bool
        insert_start : int or None
        insert_end : int or None
        warnings : list of str
    """
    prod = _validate_dna(product, "product")
    ins = _validate_dna(expected_insert, "insert")
    warnings = []

    # Check forward orientation
    fwd_pos = prod.find(ins)
    # Check reverse orientation
    rev_ins = _reverse_complement(ins)
    rev_pos = prod.find(rev_ins)

    insert_found = fwd_pos != -1 or rev_pos != -1

    if fwd_pos != -1 and rev_pos != -1:
        # Found in both orientations — check which is more likely
        warnings.append(
            "Insert found in BOTH orientations. Verify by restriction mapping."
        )
        forward_orientation = True
        insert_start = fwd_pos
        insert_end = fwd_pos + len(ins)
    elif fwd_pos != -1:
        forward_orientation = True
        insert_start = fwd_pos
        insert_end = fwd_pos + len(ins)
    elif rev_pos != -1:
        forward_orientation = False
        insert_start = rev_pos
        insert_end = rev_pos + len(rev_ins)
    else:
        forward_orientation = False
        insert_start = None
        insert_end = None
        warnings.append("Insert NOT found in the product sequence.")

    # Check for partial matches (truncated insert)
    if not insert_found and len(ins) > 20:
        half = len(ins) // 2
        front = ins[:half]
        back = ins[half:]
        if front in prod:
            warnings.append(
                f"Only the 5' half of the insert (first {half} bp) was found. "
                "Insert may be truncated."
            )
        elif back in prod:
            warnings.append(
                f"Only the 3' half of the insert (last {half} bp) was found. "
                "Insert may be truncated."
            )

    return {
        "insert_found": insert_found,
        "forward_orientation": forward_orientation,
        "insert_start": insert_start,
        "insert_end": insert_end,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Virtual gel electrophoresis
# ---------------------------------------------------------------------------
def plot_virtual_gel(fragment_sizes: List[int],
                     ladder: Optional[List[int]] = None,
                     ax: Optional[plt.Axes] = None,
                     title: str = "Virtual Gel",
                     lane_labels: Optional[List[str]] = None,
                     sample_colors: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot a virtual agarose gel with multiple sample lanes support.

    Parameters
    ----------
    fragment_sizes : list of int
        Sample fragment sizes in bp.
    ladder : list of int, optional
        Molecular weight ladder sizes.
    ax : matplotlib Axes, optional
        If provided, draw onto these axes.
    title : str
        Plot title.
    lane_labels : list of str, optional
        Labels for each sample lane.
    sample_colors : list of str, optional
        Colors for each sample lane (e.g., ['cyan', 'magenta']).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ladder is None:
        ladder = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 700, 500, 200]

    # Handle fragment_sizes as list of lists for multi-lane support
    if fragment_sizes and isinstance(fragment_sizes[0], (list, tuple)):
        all_samples = [list(s) for s in fragment_sizes]
    else:
        all_samples = [list(fragment_sizes)]

    n_samples = len(all_samples)
    total_lanes = 1 + n_samples  # ladder + samples

    if ax is None:
        fig, ax = plt.subplots(figsize=(4 + 2 * n_samples, 8))
    else:
        fig = ax.get_figure()

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    gel_left, gel_right = 0.10, 0.90
    gel_top, gel_bottom = 0.88, 0.08
    lane_width = (gel_right - gel_left) / total_lanes

    gel_rect = plt.Rectangle((gel_left, gel_bottom), gel_right - gel_left,
                              gel_top - gel_bottom, linewidth=2,
                              edgecolor="white", facecolor="#1a1a2e", zorder=1)
    ax.add_patch(gel_rect)

    # Collect all sizes for scale (filter out invalid values)
    all_valid_ladder = [s for s in ladder if isinstance(s, (int, float)) and s > 0]
    all_valid_samples = [s for sample in all_samples
                         for s in sample
                         if isinstance(s, (int, float)) and s > 0]

    all_sizes = sorted(set(all_valid_ladder + all_valid_samples), reverse=True)

    if not all_sizes:
        all_sizes = [100]

    max_size = max(all_sizes)
    min_size = min(all_sizes)
    log_max = math.log10(max_size) if max_size > 0 else 1.0
    log_min = math.log10(min_size) if min_size > 0 else 0.0

    def size_to_y(size):
        """Map fragment size to vertical position on the gel.
        Uses log scale. Clamps to gel bounds for out-of-range sizes."""
        if size <= 0:
            return gel_bottom
        try:
            log_size = math.log10(size)
        except (ValueError, ZeroDivisionError):
            return gel_bottom

        if log_max == log_min:
            frac = 0.5
        else:
            frac = (log_size - log_min) / (log_max - log_min)

        # Clamp to gel bounds
        frac = max(0.0, min(1.0, frac))
        return gel_top - frac * (gel_top - gel_bottom)

    def band_intensity(size, is_ladder=False):
        """Calculate band intensity based on size and type."""
        if size <= 0:
            return 0.0
        try:
            log_ratio = math.log10(size) / log_max if log_max > 0 else 0.5
        except (ValueError, ZeroDivisionError):
            return 0.5
        if is_ladder:
            return min(1.0, max(0.2, 0.3 + 0.7 * log_ratio))
        return min(1.0, max(0.2, 0.3 + 0.6 * log_ratio))

    # ── Ladder (lane 0) ──────────────────────────────────────────────────
    ladder_x = gel_left + lane_width * 0.5
    for size in all_valid_ladder:
        y = size_to_y(size)
        intensity = band_intensity(size, is_ladder=True)
        color = (0.2 * intensity, 0.9 * intensity, 0.3 * intensity)
        glow = (0.1, 1.0, 0.2, 0.15)

        # Glow halo
        ax.fill_between([ladder_x - lane_width * 0.3, ladder_x + lane_width * 0.3],
                        y - 0.012, y + 0.012, color=glow, zorder=2)
        # Band
        ax.fill_between([ladder_x - lane_width * 0.18, ladder_x + lane_width * 0.18],
                        y - 0.004, y + 0.004, color=color, zorder=3)
        # Label
        ax.text(ladder_x - lane_width * 0.35, y, f"{size}", fontsize=5.5,
                color="white", ha="right", va="center", fontfamily="monospace", zorder=4)

    # ── Sample lanes ──────────────────────────────────────────────────────
    default_colors = [
        ((0.15, 0.85, 0.25), (0.05, 0.8, 0.15, 0.18)),   # green
        ((0.85, 0.15, 0.25), (0.8, 0.05, 0.15, 0.18)),   # red
        ((0.15, 0.45, 0.85), (0.05, 0.35, 0.8, 0.18)),   # blue
        ((0.85, 0.85, 0.15), (0.8, 0.8, 0.05, 0.18)),    # yellow
        ((0.85, 0.15, 0.85), (0.8, 0.05, 0.8, 0.18)),    # magenta
    ]

    for lane_idx, sample_frags in enumerate(all_samples):
        lane_x = gel_left + lane_width * (lane_idx + 0.5)

        # Determine color
        if sample_colors and lane_idx < len(sample_colors):
            band_color_name = sample_colors[lane_idx]
            # Map color name to RGB
            color_map = {
                "green": ((0.15, 0.85, 0.25), (0.05, 0.8, 0.15, 0.18)),
                "red": ((0.85, 0.15, 0.25), (0.8, 0.05, 0.15, 0.18)),
                "blue": ((0.15, 0.45, 0.85), (0.05, 0.35, 0.8, 0.18)),
                "cyan": ((0.15, 0.85, 0.85), (0.05, 0.8, 0.8, 0.18)),
                "yellow": ((0.85, 0.85, 0.15), (0.8, 0.8, 0.05, 0.18)),
                "magenta": ((0.85, 0.15, 0.85), (0.8, 0.05, 0.8, 0.18)),
                "white": ((0.9, 0.9, 0.9), (0.8, 0.8, 0.8, 0.18)),
            }
            color_pair = color_map.get(band_color_name.lower(),
                                        default_colors[lane_idx % len(default_colors)])
        else:
            color_pair = default_colors[lane_idx % len(default_colors)]

        band_color, glow = color_pair

        # Filter valid fragment sizes for this sample
        valid_frags = [s for s in sample_frags
                       if isinstance(s, (int, float)) and s > 0]

        for size in valid_frags:
            y = size_to_y(size)
            intensity = band_intensity(size)

            # Apply intensity to color
            adj_color = tuple(c * intensity for c in band_color)

            # Glow
            ax.fill_between([lane_x - lane_width * 0.35, lane_x + lane_width * 0.35],
                            y - 0.018, y + 0.018, color=glow, zorder=2)
            # Band
            ax.fill_between([lane_x - lane_width * 0.22, lane_x + lane_width * 0.22],
                            y - 0.005, y + 0.005, color=adj_color, zorder=3)
            # Label
            ax.text(lane_x + lane_width * 0.28, y, f"{int(size)} bp", fontsize=5.5,
                    color="white", ha="left", va="center", fontfamily="monospace", zorder=4)

        # Lane label
        label = (lane_labels[lane_idx] if lane_labels and lane_idx < len(lane_labels)
                 else f"Sample {lane_idx + 1}")
        ax.text(lane_x, gel_top + 0.03, label, fontsize=8, color="white",
                ha="center", va="bottom", fontweight="bold")

    # Ladder label
    ax.text(ladder_x, gel_top + 0.03, "Ladder", fontsize=8, color="white",
            ha="center", va="bottom", fontweight="bold")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=12)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def format_digest_report(result: dict) -> str:
    """Return a formatted text report for a digestion result."""
    lines = [
        "=" * 60,
        "  RESTRICTION DIGESTION REPORT",
        "=" * 60,
        f"  Total input: {result['total_bp']:,} bp",
        f"  Enzymes: {', '.join(result['enzymes_used'])}",
        "-" * 60,
    ]

    # Overhang types
    overhangs = result.get("overhangs", {})
    if overhangs:
        lines.append("  Enzyme overhangs:")
        for enz, oh_type in overhangs.items():
            lines.append(f"    {enz}: {oh_type}")
        lines.append("-" * 60)

    if result["cuts"]:
        lines.append(f"  Cut sites ({len(result['cuts'])}):")
        for cut in result["cuts"]:
            if isinstance(cut, dict):
                lines.append(
                    f"    Position {cut['cut_pos']:,} "
                    f"({cut['enzyme']}, offset +{cut['cut_offset']})"
                )
            else:
                lines.append(f"    Position {cut:,}")
    else:
        lines.append("  No cut sites found.")

    lines.append("-" * 60)
    lines.append(f"  Fragments: {len(result['sizes'])}")
    lines.append(f"  {'#':<4} {'Size (bp)':>10}")
    lines.append(f"  {'—'*4} {'—'*10}")
    for i, size in enumerate(sorted(result["sizes"], reverse=True), 1):
        lines.append(f"  {i:<4} {size:>10,}")
    lines.append(f"  {'—'*4} {'—'*10}")
    lines.append(f"  {'Total':<4} {sum(result['sizes']):>10,}")

    # Sanity check: total should match
    if sum(result["sizes"]) != result["total_bp"]:
        lines.append(f"  ⚠ WARNING: Fragment total ({sum(result['sizes']):,}) "
                     f"≠ input ({result['total_bp']:,})")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_primer_report(primers: dict) -> str:
    """Return a formatted text report for a primer pair dict."""
    fwd = primers["forward"]
    rev = primers["reverse"]
    fwd_tm = primers.get("fwd_tm", 0)
    rev_tm = primers.get("rev_tm", 0)
    fwd_gc = primers.get("fwd_gc", 0)
    rev_gc = primers.get("rev_gc", 0)
    warnings = primers.get("warnings", [])

    lines = [
        "=" * 60,
        "  PRIMER DESIGN REPORT",
        "=" * 60,
        "",
        f"  Forward Primer: 5'-{fwd}-3'",
        f"    Length: {len(fwd)} nt   Tm: {fwd_tm:.1f} °C   GC: {fwd_gc:.1f}%",
        "",
        f"  Reverse Primer: 5'-{rev}-3'",
        f"    Length: {len(rev)} nt   Tm: {rev_tm:.1f} °C   GC: {rev_gc:.1f}%",
        "",
    ]

    delta = abs(fwd_tm - rev_tm)
    lines.append(f"  ΔTm: {delta:.1f} °C")
    lines.append("  ✓ Good match" if delta <= 5 else "  ⚠ Large Tm difference")

    if warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in warnings:
            lines.append(f"    ⚠ {w}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_pcr_report(result: dict) -> str:
    """Return a formatted text report for a PCR result."""
    lines = [
        "=" * 60,
        "  PCR AMPLIFICATION REPORT",
        "=" * 60,
        f"  Template length: N/A (see product size)",
        f"  Product size: {result['size']:,} bp",
        f"  Cycles: {result['cycles']}",
        f"  Estimated efficiency: {result.get('efficiency', 0) * 100:.0f}%",
        f"  Forward primer at: position {result.get('fwd_pos', 'N/A')}",
        f"  Reverse primer RC at: position {result.get('rev_pos', 'N/A')}",
        "-" * 60,
    ]

    warnings = result.get("warnings", [])
    if warnings:
        lines.append("  Warnings:")
        for w in warnings:
            lines.append(f"    ⚠ {w}")
    else:
        lines.append("  ✓ No warnings")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_ligation_report(result: dict) -> str:
    """Return a formatted text report for a ligation result."""
    lines = [
        "=" * 60,
        "  LIGATION REPORT",
        "=" * 60,
        f"  Fragments ligated: {result.get('fragment_count', 0)}",
        f"  Product size: {result.get('size', 0):,} bp",
        f"  Topology: {'circular' if result.get('is_circular') else 'linear'}",
        f"  Estimated efficiency: {result.get('efficiency', 0) * 100:.1f}%",
    ]

    verified = result.get("verified")
    if verified is True:
        lines.append("  ✓ Insert verified in product")
    elif verified is False and result.get("fragment_count", 0) > 0:
        lines.append("  — Insert verification: not requested or no insert specified")

    warnings = result.get("warnings", [])
    if warnings:
        lines.append("-" * 60)
        lines.append("  Notes:")
        for w in warnings:
            lines.append(f"    ⚠ {w}")

    lines.append("=" * 60)
    return "\n".join(lines)
