"""
Molecular Cloning Toolkit — restriction digestion, ligation, PCR, and
virtual gel electrophoresis.

Implemented with matplotlib only — no numpy and no Biopython are used here.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple, Union

# No matplotlib.use() here: a library module must not replace the host
# process' backend (F2-4).  Headless callers declare that themselves.
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ---------------------------------------------------------------------------
# Restriction enzyme database (imported from shared utils)
# ---------------------------------------------------------------------------
# We need BOTH the sites and the cut positions for accurate digestion.
from biosuite.core.utils import RESTRICTION_ENZYMES, enzyme_cut_type

# ---------------------------------------------------------------------------
# DNA validation helper
# ---------------------------------------------------------------------------
_VALID_DNA = set("ATCGWSMKRYBDHVN")


def _validate_dna(seq: str, name: str = "sequence") -> str:
    """Validate and clean a DNA sequence. Raises ValueError on bad input.

    "Clean" means upper-cased with every whitespace removed, because a sequence
    pasted out of a FASTA record arrives in 60-column lines.  Only the ends were
    stripped before, so the line breaks of a multi-line sequence were reported
    back as ``Invalid characters in <name>:`` followed by an invisible newline.
    """
    seq = re.sub(r"\s+", "", seq.upper())
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
                             "TAGCWSKMYRVHDBN")

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
        return _find_all_sites(seq, site)

    # Return per-enzyme results
    results: Dict[str, List[int]] = {}
    for name, (site, _cut) in RESTRICTION_ENZYMES.items():
        hits = _find_all_sites(seq, site)
        if hits:
            results[name] = hits
    return results


# Every IUPAC nucleotide code and the concrete bases it stands for.  This is the
# single source: the recognition-site matcher reads it for a character class per
# code, and the primer matcher reads it to compare two codes as base sets.
_IUPAC_BASES = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}


def _code_class(code: str) -> str:
    """One position of a pattern, as the regex fragment that matches it.

    A concrete base stays a literal; a degenerate code becomes the character
    class of the bases it can be.
    """
    bases = _IUPAC_BASES.get(code)
    if bases is None:
        return re.escape(code)
    return bases if len(bases) == 1 else f"[{bases}]"


def _site_regex(site: str) -> "re.Pattern[str]":
    """Compile a recognition site so each IUPAC code matches the bases it stands for.

    Wrapped in a lookahead so overlapping sites are all reported — a plain
    ``finditer`` over the bare body would silently drop them.

    The template is read literally: a position whose base is unknown does not
    complete a recognition site, so an ambiguous stretch never reports a site.
    """
    body = "".join(_code_class(ch) for ch in site.upper())
    return re.compile("(?=" + body + ")")


def _binding_regex(primer: str) -> "re.Pattern[str]":
    """Compile a primer so each position matches a compatible template base.

    A consensus primer is a *set of sequences* by convention (``R`` means the
    primer population carries A or G there), so it binds wherever one of its
    instantiations occurs.  The template position may itself be ambiguous, so the
    two codes are compared as base sets and any shared base is a binding site.
    """
    body = ""
    for ch in primer.upper():
        wanted = set(_IUPAC_BASES.get(ch, ch))
        codes = [code for code, bases in _IUPAC_BASES.items()
                 if wanted & set(bases)]
        # A letter outside the IUPAC alphabet can only ever match itself.
        body += re.escape(ch) if not codes else "[" + "".join(sorted(codes)) + "]"
    return re.compile("(?=" + body + ")")


def _binding_sites(seq: str, primer: str) -> List[int]:
    """Every position where *primer* can anneal to *seq* (overlapping)."""
    return [m.start() for m in _binding_regex(primer).finditer(seq)]


def _occurrences(hay: str, needle: str) -> List[int]:
    """Every start position of the exact string *needle* in *hay*, overlaps included.

    Callers pass an insert that _validate_dna() already proved non-empty; an
    empty needle would report a match at every position.
    """
    positions = []
    i = hay.find(needle)
    while i != -1:
        positions.append(i)
        i = hay.find(needle, i + 1)
    return positions


def _find_all(seq: str, pattern: str) -> List[int]:
    """Find all start positions of *pattern* in *seq* (overlapping)."""
    return [m.start() for m in _site_regex(pattern).finditer(seq)]


def _site_and_rc(site: str) -> Tuple[str, str]:
    """Return the recognition site and its reverse complement, upper-cased.

    They differ for non-palindromic enzymes (Type IIS among them), which bind a
    site read on the opposite strand just as readily.
    """
    upper = site.upper()
    return upper, _reverse_complement(upper)


def _find_all_sites(seq: str, site: str) -> List[int]:
    """Every position where *site* is recognised, on either strand."""
    forward, reverse = _site_and_rc(site)
    hits = _find_all(seq, forward)
    if reverse != forward:
        hits = sorted(set(hits) | set(_find_all(seq, reverse)))
    return hits


# ---------------------------------------------------------------------------
# Restriction digestion
# ---------------------------------------------------------------------------
TOPOLOGIES = ("circular", "linear")


def _resolve_topology(topology: str) -> str:
    """Return the canonical spelling of *topology* or refuse it.

    The molecule's shape decides the fragment count, so a value that is merely
    close to the right word ("Circular", "circle") must not be silently read as
    the other one -- it used to be, because only the exact string compared equal.
    """
    canonical = topology.strip().lower() if isinstance(topology, str) else topology
    if canonical not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology: {topology!r}. Use 'circular' (a plasmid) or "
            "'linear' (a PCR product or genomic fragment).")
    return canonical


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
        "circular" or "linear"; any other spelling is refused.  A case or space
        difference is accepted (" Circular " reads as circular).

    Returns
    -------
    dict with keys:
        fragments : list of dict  — each dict has 'size', 'sequence', 'start', 'end'
        sizes : list of int       — fragment sizes in bp (for backward compat)
        total_bp : int            — total sequence length
        cuts : list of dict       — cut info dicts {enzyme, site_start, cut_offset,
                                                    cut_pos, strand}
        enzymes_used : list       — enzyme names used
        overhangs : dict          — enzyme_name -> overhang type ('5prime', '3prime', 'blunt')
        warnings : list of str    — caveats; non-empty when an enzyme's cut geometry
                                    was derived from an unverified table entry

    Raises
    ------
    ValueError
        If *enzyme* or *topology* is not usable.
    """
    seq = _validate_dna(sequence, "digestion template")
    topo = _resolve_topology(topology)
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
    # A circular molecule can present a recognition site that straddles the origin,
    # so the search text is extended by the longest site minus one base and hits
    # beyond the real length are dropped.
    search_seq = seq
    if topo == "circular" and enzyme_info:
        longest = max(len(enz_site) for enz_site, _offset in enzyme_info.values())
        search_seq = seq + seq[:max(longest - 1, 0)]

    cut_data: List[Dict] = []  # [{enzyme, site_start, cut_offset, cut_pos, strand}]
    asymmetric: List[str] = []
    for enz_name, (site, cut_offset) in enzyme_info.items():
        forward, reverse = _site_and_rc(site)
        occurrences = [(pos, "+") for pos in _find_all(search_seq, forward) if pos < total]
        if reverse != forward:
            occurrences += [(pos, "-") for pos in _find_all(search_seq, reverse) if pos < total]
        if reverse != forward and occurrences:
            # Only an enzyme that actually cut has a cut position to approximate.
            asymmetric.append(enz_name)
        for pos, strand in sorted(occurrences):
            cut_pos = pos + cut_offset
            if topo == "circular":
                cut_pos %= total
            cut_data.append({
                "enzyme": enz_name,
                "site_start": pos,
                "cut_offset": cut_offset,
                "cut_pos": cut_pos,
                "strand": strand,
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

    # Classify overhang types: REBASE-verified geometry wins, otherwise derive it
    # from the stored offset and record that the result is only a guess.
    overhangs: Dict[str, str] = {}
    unverified_geometry: List[str] = []
    for enz_name, (site, cut_offset) in enzyme_info.items():
        verified = enzyme_cut_type(enz_name)
        if verified != "unverified":
            overhangs[enz_name] = verified
            continue
        overhang = len(site) - 2 * cut_offset
        overhangs[enz_name] = ("blunt" if overhang == 0
                               else "5prime" if overhang > 0 else "3prime")
        unverified_geometry.append(enz_name)

    digest_warnings: List[str] = []
    if unverified_geometry:
        digest_warnings.append(
            "cut geometry derived from unchecked table entries for: "
            + ", ".join(sorted(unverified_geometry)))
    if asymmetric:
        digest_warnings.append(
            "non-palindromic site recognised on either strand, but the double-strand "
            "cut geometry of such enzymes is not modelled, so cut_pos is a top-strand "
            "approximation for: " + ", ".join(sorted(asymmetric)))

    if not cut_data:
        return {
            "fragments": [{"size": total, "sequence": seq,
                           "start": 0, "end": total}],
            "sizes": [total],
            "total_bp": total,
            "cuts": [],
            "enzymes_used": enzyme_list,
            "overhangs": overhangs,
            "warnings": digest_warnings,
        }

    cut_positions = [c["cut_pos"] for c in cut_data]

    # Generate fragments
    if topo == "circular":
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
        # Linear: pieces between cuts + ends.  Every table enzyme has
        # cut_offset < len(site) and a hit is only recorded when the whole site
        # fits in the molecule, so 0 < cut_pos < total and the deduplicated,
        # sorted cuts are strictly increasing: no piece can be empty.
        fragments = []
        positions = [0] + cut_positions + [total]
        for i in range(len(positions) - 1):
            s, e = positions[i], positions[i + 1]
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
        "warnings": digest_warnings,
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
        DNA fragment sequences to ligate.  One fragment is ``[sequence]``; a bare
        sequence is refused rather than split into single bases.
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

    if isinstance(fragments, str):
        # Iterating a bare sequence yields single bases, every one of which is
        # valid DNA, so the old code "ligated" the template into N 1-nt
        # fragments and reported an efficiency for a ligation that never happened.
        raise ValueError(
            "fragments must be a list of fragment sequences, not one bare "
            f"sequence ({len(fragments)} nt was read as {len(fragments)} "
            "1-nt fragments). Pass [sequence] for a single fragment.")

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
            _validate_dna(vector_sequence, "vector")   # validate, value unused
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
    """Melting temperature from base counts only.

    Below 14 nt the Wallace rule (2 °C per A/T, 4 °C per G/C) is used; at 14 nt
    and above the basic composition formula Tm = 64.9 + 41·(nGC − 16.4)/L.  This
    is *not* a nearest-neighbour calculation: two primers with the same counts in
    a different order get the same value, so base order and salt concentration
    are not modelled.  For a primer pair of equal length the bias is the same on
    both sides, which is why the ΔTm comparison in design_primers() stays useful
    even though the absolute numbers read low for short oligonucleotides.
    """
    s = seq.upper()
    nA = s.count("A"); nT = s.count("T")
    nG = s.count("G"); nC = s.count("C")
    length = nA + nT + nG + nC
    if length == 0:
        return 0.0
    if length < 14:
        # Wallace rule (approximate)
        return 2.0 * (nA + nT) + 4.0 * (nG + nC)
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

    if primer_length < 1:
        # seq[:0] is empty while seq[-0:] is the *whole* template, so a
        # non-positive length used to return an empty forward primer and a
        # "reverse" primer as long as the target -- and a negative one silently
        # trimmed both ends instead.
        raise ValueError(
            f"Primer length must be at least 1 nt, got {primer_length}.")

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

    Primers are matched position by position, so a consensus primer written with
    IUPAC codes (``R`` = A or G) anneals wherever one of its instantiations
    occurs, and a template position that is itself ambiguous accepts any primer
    code sharing a base with it.  When either primer matches more than once, the
    product is the first forward site that has a reverse-primer site downstream
    of it, paired with the nearest such site.

    Returns
    -------
    dict with keys:
        product : str      — amplified DNA sequence
        size : int         — product length in bp
        cycles : int       — number of cycles used
        fwd_pos : int      — forward primer binding position (0-based)
        rev_pos : int      — start of the reverse primer's binding site on the
                             top strand (0-based)
        efficiency : float — estimated per-cycle amplification efficiency
        warnings : list of str

    Raises
    ------
    ValueError
        If a primer cannot be found on the template, or if no reverse site lies
        downstream of a forward site.
    """
    seq = _validate_dna(template, "PCR template")
    fwd = _validate_dna(fwd_primer, "forward primer")
    rev = _validate_dna(rev_primer, "reverse primer")
    warnings = []

    if cycles < 1 or cycles > 50:
        warnings.append(f"Unusual cycle count: {cycles}. Typical range is 25-35.")

    fwd_sites = _binding_sites(seq, fwd)
    if not fwd_sites:
        # The 3' 6 bases are the critical ones, so a template carrying them is
        # a near miss.  The hint travels in the exception: this branch raises,
        # so a warning appended to the local list would be discarded with it.
        hint = ""
        if len(fwd) > 6:
            partial = _binding_sites(seq, fwd[-6:])
            if partial:
                hint = (f" Closest partial match (3' 6 bases) at position "
                        f"{partial[0] - len(fwd) + 6}.")
        raise ValueError(
            f"Forward primer '{fwd}' not found in template.{hint} "
            "Check primer sequence and template."
        )

    # Reverse primer binds to the opposite strand → find its RC on the template
    rev_rc = _reverse_complement(rev)
    rev_sites = _binding_sites(seq, rev_rc)
    if not rev_sites:
        hint = ""
        if len(rev_rc) > 6:
            partial = _binding_sites(seq, rev_rc[-6:])
            if partial:
                hint = (f" Closest partial match (3' 6 bases) at position "
                        f"{partial[0] - len(rev_rc) + 6}.")
        raise ValueError(
            f"Reverse primer binding site (RC: '{rev_rc}') not found in template."
            f"{hint} Check primer sequence and template."
        )

    # Validate primer orientation: forward must bind before reverse.  A primer
    # can anneal at more than one place, so the product is defined by the first
    # forward site that has a reverse site downstream of it -- reading only the
    # leftmost reverse site used to reject a pair that would amplify normally.
    pairing = next(((f, r) for f in fwd_sites for r in rev_sites if r > f), None)
    if pairing is None:
        raise ValueError(
            f"Primer orientation error: forward primer binds at position {fwd_sites[0]} "
            f"but reverse primer RC binds at position {rev_sites[0]}. "
            "Forward primer must bind upstream of the reverse primer binding site."
        )
    fwd_pos, rev_pos = pairing

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
        If provided, checks that the insert sits between vector material: the
        product with the insert cut out must still read as a contiguous run of
        the (circular) backbone.  A warning is added when it does not, which is
        what a primer that amplified vector instead of insert, or a backbone
        handed in from a different construct, produces.

    Returns
    -------
    dict with keys:
        insert_found : bool
        forward_orientation : bool
                             Always True when a self-complementary insert is
                             found: its two readings are the same string.
        insert_start : int or None
        insert_end : int or None
                    End of the located copy plus any head-to-tail repeats of it,
                    so a tandem insertion is reported as one inserted block.
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

    if ins == rev_ins:
        # A self-complementary insert is the same string read either way, so
        # finding it reports one occurrence, not "both orientations".
        if fwd_pos != -1:
            forward_orientation = True
            insert_start = fwd_pos
            insert_end = fwd_pos + len(ins)
            warnings.append(
                "The insert is self-complementary, so its orientation in the "
                "product cannot be distinguished."
            )
        else:
            forward_orientation = False
            insert_start = None
            insert_end = None
            warnings.append("Insert NOT found in the product sequence.")
    elif fwd_pos != -1 and rev_pos != -1:
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

    if insert_found:
        # A tandem repeat of the insert is an ordinary cloning outcome, and the
        # backbone test below would otherwise read the second copy as extra
        # vector material.
        occurrences = _occurrences(prod, ins)
        if rev_ins != ins:
            occurrences += _occurrences(prod, rev_ins)
        if len(occurrences) > 1:
            shown = ", ".join(str(p) for p in sorted(occurrences)[:8])
            more = ", ..." if len(occurrences) > 8 else ""
            block = ins if forward_orientation else rev_ins
            while prod[insert_end:insert_end + len(block)] == block:
                insert_end += len(block)
            warnings.append(
                f"The insert occurs {len(occurrences)} times in the product "
                f"(at {shown}{more}). insert_start/insert_end span the first "
                "copy and any copies fused to it head-to-tail."
            )

    # Check for partial matches (truncated insert)
    if not insert_found and len(ins) > 20:
        half = len(ins) // 2
        candidates = (
            (ins[:half], f"5' half of the insert (first {half} bp)", ""),
            (ins[half:], f"3' half of the insert (last {half} bp)", ""),
            (rev_ins[:half],
             f"5' half of the insert read backwards (first {half} bp)",
             " and reverse-oriented"),
            (rev_ins[half:],
             f"3' half of the insert read backwards (last {half} bp)",
             " and reverse-oriented"),
        )
        for piece, label, extra in candidates:
            if piece in prod:
                warnings.append(
                    f"Only the {label} was found. Insert may be truncated{extra}."
                )
                break

    if vector_backbone is not None and insert_found:
        backbone = _validate_dna(vector_backbone, "vector_backbone")
        # The product is the backbone opened at one site with the insert dropped
        # in, so cutting the insert back out must leave a contiguous run of the
        # circular backbone.  Searching the doubled backbone finds that rotation
        # without assuming where the cut was made.
        retained = prod[:insert_start] + prod[insert_end:]
        if not retained:
            warnings.append(
                "No backbone material remains outside the insert: the product is "
                "the insert alone.")
        elif len(retained) > len(backbone):
            warnings.append(
                f"Vector material outside the insert is {len(retained)} bp but the "
                f"backbone is {len(backbone)} bp: the product carries more vector "
                "than was handed in.")
        elif (backbone + backbone).find(retained) < 0:
            warnings.append(
                "The sequence flanking the insert is not contiguous in the given "
                "backbone: the insert is not between vector segments.")

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
# House palette for the named lanes: (band RGB, glow RGBA).  Anything outside it
# is resolved by matplotlib, so a hex triplet or a matplotlib colour name works.
_PRESET_LANE_COLORS: Dict[str, Tuple[Tuple[float, float, float],
                                     Tuple[float, float, float, float]]] = {
    "green": ((0.15, 0.85, 0.25), (0.05, 0.8, 0.15, 0.18)),
    "red": ((0.85, 0.15, 0.25), (0.8, 0.05, 0.15, 0.18)),
    "blue": ((0.15, 0.45, 0.85), (0.05, 0.35, 0.8, 0.18)),
    "cyan": ((0.15, 0.85, 0.85), (0.05, 0.8, 0.8, 0.18)),
    "yellow": ((0.85, 0.85, 0.15), (0.8, 0.8, 0.05, 0.18)),
    "magenta": ((0.85, 0.15, 0.85), (0.8, 0.05, 0.8, 0.18)),
    "white": ((0.9, 0.9, 0.9), (0.8, 0.8, 0.8, 0.18)),
}

_DEFAULT_LANE_COLORS = [
    _PRESET_LANE_COLORS["green"],
    _PRESET_LANE_COLORS["red"],
    _PRESET_LANE_COLORS["blue"],
    _PRESET_LANE_COLORS["yellow"],
    _PRESET_LANE_COLORS["magenta"],
]


def _lane_palette(name: Union[str, Tuple[float, ...]]) -> Tuple[Tuple[float, float, float],
                                                                 Tuple[float, float, float, float]]:
    """Resolve one lane colour into its (band RGB, glow RGBA) pair.

    A colour that neither the house palette nor matplotlib can parse raises:
    it used to fall back to the lane's default colour, so a misspelt name drew a
    confident gel in a colour nobody asked for.
    """
    if isinstance(name, str) and name.lower() in _PRESET_LANE_COLORS:
        return _PRESET_LANE_COLORS[name.lower()]
    try:
        band = to_rgba(name)[:3]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unusable lane colour: {name!r}") from exc
    return band, band + (0.18,)


def _as_lane_list(value, name: str):
    """Wrap a single label or colour so a bare string is not read as a sequence."""
    if isinstance(value, str):
        return [value]
    if value is None or isinstance(value, (list, tuple)):
        return value
    raise ValueError(f"{name} must be a list or a single string, got {type(value).__name__}")


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
    fragment_sizes : list of int, or list of lists
        Sample fragment sizes in bp — a flat list is one lane, a list of lists
        is one lane per inner list.  Digest-style dicts carrying a 'size' key are
        accepted in either, and an entry with no usable size is dropped.
    ladder : list of int, optional
        Molecular weight ladder sizes.
    ax : matplotlib Axes, optional
        If provided, draw onto these axes.
    title : str
        Plot title.
    lane_labels : list of str, optional
        Labels for each sample lane.  One lane may be labelled with a bare
        string.
    sample_colors : list of str, optional
        Colors for each sample lane (e.g., ['cyan', 'magenta']).  Names from the
        house palette, any matplotlib colour name, a hex triplet or an RGB tuple
        are accepted; a colour that cannot be parsed raises.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ladder is None:
        ladder = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 700, 500, 200]

    # Handle fragment_sizes as list of lists for multi-lane support
    # Also accept digest-style fragment dicts ({'size', 'sequence', ...})
    # so callers can pass simulate_digestion() output directly.
    def _norm_frags(frags):
        out = []
        for f in frags:
            if isinstance(f, dict):
                size = f.get('size')
                if isinstance(size, (int, float)):
                    out.append(size)
            else:
                out.append(f)
        return out

    lane_labels = _as_lane_list(lane_labels, "lane_labels")
    sample_colors = _as_lane_list(sample_colors, "sample_colors")

    if fragment_sizes and isinstance(fragment_sizes[0], (list, tuple)):
        # One lane per inner list.  A flat value mixed into that shape used to
        # crash as ``TypeError: 'int' object is not iterable``.
        all_samples = []
        for lane_number, lane in enumerate(fragment_sizes, 1):
            if not isinstance(lane, (list, tuple)):
                raise ValueError(
                    f"lane {lane_number} of fragment_sizes is {lane!r}, not a list "
                    "of fragment sizes. A multi-lane gel takes a list of lists; a "
                    "single sample takes a flat list.")
            all_samples.append(_norm_frags(list(lane)))
    else:
        all_samples = [_norm_frags(list(fragment_sizes))]

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
        Uses log scale. Clamps to gel bounds for out-of-range sizes.

        Both call sites pass a size already filtered by ``isinstance(s, (int,
        float)) and s > 0``, so log10() cannot raise and a non-positive size
        never reaches this mapping."""
        log_size = math.log10(size)
        if log_max == log_min:
            frac = 0.5
        else:
            frac = (log_size - log_min) / (log_max - log_min)

        # Clamp to gel bounds
        frac = max(0.0, min(1.0, frac))
        return gel_top - frac * (gel_top - gel_bottom)

    def band_intensity(size, is_ladder=False):
        """Calculate band intensity based on size and type.

        Same guarantee as size_to_y: *size* is > 0 here, so only log_max can
        still be <= 0 (a gel whose every size is below 1 bp)."""
        log_ratio = math.log10(size) / log_max if log_max > 0 else 0.5
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
    for lane_idx, sample_frags in enumerate(all_samples):
        # Offset by +1.5: lane 0 belongs to the ladder — sample lanes
        # start after it (previously they overlapped the ladder lane).
        lane_x = gel_left + lane_width * (lane_idx + 1.5)

        # Determine color
        if sample_colors and lane_idx < len(sample_colors):
            color_pair = _lane_palette(sample_colors[lane_idx])
        else:
            color_pair = _DEFAULT_LANE_COLORS[lane_idx % len(_DEFAULT_LANE_COLORS)]

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
    """Return a formatted text report for a digestion result.

    *result* is the dict from simulate_digestion(); 'overhangs' and 'warnings'
    may be left out because both describe optional state, everything else is
    read directly -- a report must not print a zero for a number it never got.
    """
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

    for note in result.get("warnings", []):
        lines.append(f"  ! {note}")
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
    """Return a formatted text report for a primer pair dict.

    The four measurements are read directly: a dict that lacks them used to be
    reported as a primer pair with Tm 0.0 °C and GC 0.0 %, which is a number the
    caller never measured.
    """
    fwd = primers["forward"]
    rev = primers["reverse"]
    fwd_tm = primers["fwd_tm"]
    rev_tm = primers["rev_tm"]
    fwd_gc = primers["fwd_gc"]
    rev_gc = primers["rev_gc"]
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
        "  Template length: N/A (see product size)",
        f"  Product size: {result['size']:,} bp",
        f"  Cycles: {result['cycles']}",
        f"  Estimated efficiency: {result['efficiency'] * 100:.0f}%",
        f"  Forward primer at: position {result['fwd_pos']}",
        f"  Reverse primer RC at: position {result['rev_pos']}",
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
        f"  Fragments ligated: {result['fragment_count']}",
        f"  Product size: {result['size']:,} bp",
        f"  Topology: {'circular' if result['is_circular'] else 'linear'}",
        f"  Estimated efficiency: {result['efficiency'] * 100:.1f}%",
    ]

    verified = result["verified"]
    if verified is True:
        lines.append("  ✓ Insert verified in product")
    elif verified is False and result["fragment_count"] > 0:
        lines.append("  — Insert verification: not requested or no insert specified")

    warnings = result.get("warnings", [])
    if warnings:
        lines.append("-" * 60)
        lines.append("  Notes:")
        for w in warnings:
            lines.append(f"    ⚠ {w}")

    lines.append("=" * 60)
    return "\n".join(lines)
