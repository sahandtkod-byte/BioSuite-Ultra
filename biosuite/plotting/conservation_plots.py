"""
Conservation and sequence logo visualization.
Generates information-content-weighted sequence logos and conservation bars.
Pure Python / matplotlib implementation.
"""
import numbers
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'compute_logo_heights',
    'plot_sequence_logo',
    'plot_conservation_bar',
    'plot_logo_with_conservation',
    'compute_conservation_scores',
    'plot_motif_enrichment',
]


def _checked_sequences(sequences, what="a sequence logo"):
    if isinstance(sequences, str):
        raise ValueError(f"{what} needs a list of aligned sequences, not a single string")
    try:
        sequences = list(sequences)
    except TypeError:
        raise ValueError(
            f"{what} needs an iterable of strings, got {type(sequences).__name__}") from None
    if not sequences:
        raise ValueError(f"{what} needs at least one sequence; the list is empty")
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise ValueError(f"sequence {index + 1} is not a string: {sequence!r}")
    sequences = [sequence.upper() for sequence in sequences]
    if not any(sequences):
        raise ValueError(f"{what} needs at least one non-empty sequence")
    return sequences


def _checked_alphabet(alphabet):
    if not isinstance(alphabet, str):
        raise ValueError(f"alphabet must be a string of symbols, got {type(alphabet).__name__}")
    alphabet = alphabet.upper()
    if len(alphabet) < 2:
        raise ValueError(f"alphabet needs at least two symbols, got {alphabet!r}")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"alphabet repeats a symbol: {alphabet!r}")
    return alphabet


def _checked_pseudocount(pseudocount):
    if isinstance(pseudocount, bool) or not isinstance(pseudocount, numbers.Real):
        raise ValueError(f"pseudocount must be a non-negative number, got {pseudocount!r}")
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError(
            f"pseudocount must be a non-negative finite number, got {pseudocount!r}")
    return float(pseudocount)


def _shannon_entropy(freqs):
    """Shannon entropy in bits for a probability distribution (dict or values)."""
    ent = 0.0
    vals = freqs.values() if isinstance(freqs, dict) else freqs
    for p in vals:
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def _normalize_frequencies(counts):
    """Convert counts to frequencies, return dict of char -> freq."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def compute_logo_heights(sequences, alphabet='ACGT', pseudocount=0):
    """Compute per-position letter heights for a sequence logo.

    Sequences are read case-insensitively; characters outside ``alphabet``
    (gaps, ambiguity codes) are ignored.  ``pseudocount`` adds that much prior
    weight to every alphabet letter at each column that has data, which pulls
    a conserved column's information content down from its maximum.

    Returns:
        positions: list of position indices
        heights: dict of {char: np.array of heights per position}
        total_height: conservation height per position (bits)

    Raises:
        ValueError: if the sequences or alphabet cannot describe a logo.
    """
    sequences = _checked_sequences(sequences)
    alphabet = _checked_alphabet(alphabet)
    pseudocount = _checked_pseudocount(pseudocount)

    max_len = max(len(s) for s in sequences)
    positions = list(range(max_len))

    all_heights = {c: np.zeros(max_len) for c in alphabet}
    total_height = np.zeros(max_len)

    for pos in range(max_len):
        counts = Counter()
        for seq in sequences:
            if pos < len(seq) and seq[pos] in alphabet:
                counts[seq[pos]] += 1

        if not counts:
            continue

        # Effective number of sequences at this position; the guard above already
        # proved at least one sequence contributed, so n_eff is never zero here.
        n_eff = sum(counts.values())

        if pseudocount:
            for c in alphabet:
                counts[c] += pseudocount

        freqs = _normalize_frequencies(counts)

        # Small-sample correction
        e_n = (len(alphabet) - 1) / (2 * np.log(2) * n_eff)

        # R* (observed entropy)
        r_star = _shannon_entropy(freqs)

        # Conservation height
        h = np.log2(len(alphabet)) - r_star - e_n
        h = max(0, h)
        total_height[pos] = h

        for c in alphabet:
            if c in freqs:
                all_heights[c][pos] = freqs[c] * h

    return positions, all_heights, total_height


def plot_sequence_logo(sequences, alphabet='ACGT', title="Sequence Logo",
                       figsize=(12, 4), ax=None, color_scheme=None):
    """Plot a sequence logo (information content weighted).

    Args:
        sequences: list of aligned sequences (same length or padded with '-').
        alphabet: characters to consider.
        title: plot title.
        figsize: figure size.
        ax: optional matplotlib axes.
        color_scheme: dict mapping character -> color.

    Raises:
        ValueError: if the sequences or alphabet cannot describe a logo.
    """
    if color_scheme is None:
        color_scheme = {
            'A': '#22c55e',  # green
            'C': '#3b82f6',  # blue
            'G': '#eab308',  # yellow/orange
            'T': '#ef4444',  # red
            'U': '#ef4444',  # RNA uracil
        }

    alphabet = _checked_alphabet(alphabet)
    positions, heights, _total_h = compute_logo_heights(sequences, alphabet)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    max_bits = np.log2(len(alphabet))

    for pos in positions:
        # Sort letters by height so tallest letter is on top
        letters_at_pos = []
        for c in alphabet:
            if heights[c][pos] > 0:
                letters_at_pos.append((c, heights[c][pos]))
        letters_at_pos.sort(key=lambda x: x[1])

        bottom = 0
        for c, h in letters_at_pos:
            color = color_scheme.get(c, 'gray')
            # Letter height = frequency * conservation
            # Scale x to be 0.8 wide centered on position
            ax.text(pos, bottom + h / 2, c, ha='center', va='center',
                    fontsize=max(6, min(14, 200 / len(positions))),
                    fontweight='bold', color=color)
            ax.bar(pos, h, bottom=bottom, width=0.8, color=color, edgecolor='white',
                   linewidth=0.5, align='center')
            bottom += h

    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, max_bits * 1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits")
    ax.set_title(title)
    ax.set_xticks(range(0, len(positions), max(1, len(positions) // 20)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


def plot_conservation_bar(sequences, alphabet='ACGT', title="Conservation",
                          figsize=(12, 2), ax=None, color='#00ff88'):
    """Plot a conservation score bar below a sequence logo.

    Conservation = 1 - (entropy / max_entropy).

    Raises:
        ValueError: if the sequences or alphabet cannot describe a logo.
    """
    alphabet = _checked_alphabet(alphabet)
    positions, _heights, total_h = compute_logo_heights(sequences, alphabet)
    max_bits = np.log2(len(alphabet))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    conservation = total_h / max_bits

    ax.bar(positions, conservation, color=color, edgecolor='black', linewidth=0.3)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Conservation")
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.set_xticks(range(0, len(positions), max(1, len(positions) // 20)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


def plot_logo_with_conservation(sequences, alphabet='ACGT',
                                title="Sequence Logo with Conservation",
                                figsize=(12, 6)):
    """Combined logo + conservation bar plot.

    Raises:
        ValueError: if the sequences or alphabet cannot describe a logo.
    """
    alphabet = _checked_alphabet(alphabet)
    sequences = _checked_sequences(sequences)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.1})
    plot_sequence_logo(sequences, alphabet, title=title, ax=ax1)
    plot_conservation_bar(sequences, alphabet, title="", ax=ax2)
    return fig


def compute_conservation_scores(sequences, alphabet='ACGT'):
    """Return per-position conservation scores (0-1).

    Raises:
        ValueError: if the sequences or alphabet cannot describe a logo.
    """
    alphabet = _checked_alphabet(alphabet)
    positions, _heights, total_h = compute_logo_heights(sequences, alphabet)
    max_bits = np.log2(len(alphabet))
    return list(zip(positions, total_h / max_bits))


def plot_motif_enrichment(sequences, motifs, alphabet='ACGT', title="Motif Enrichment",
                          figsize=(10, 4)):
    """Count occurrences of each motif and plot enrichment bar chart.

    Matching is case-insensitive.  A motif may only use letters from
    ``alphabet``; pass a wider alphabet (for example ``'ACGTN'``) to count a
    motif that contains an ambiguity code.

    Args:
        sequences: list of sequences.
        motifs: list of motif strings (e.g., ['ATG', 'TATAAA']).

    Raises:
        ValueError: if the sequences or motifs are empty, repeated or outside
            the alphabet.
    """
    alphabet = _checked_alphabet(alphabet)
    sequences = _checked_sequences(sequences, what="motif enrichment")

    if isinstance(motifs, str):
        raise ValueError("motifs must be a list of strings, not a single string")
    motifs = list(motifs)
    if not motifs:
        raise ValueError("plot_motif_enrichment needs at least one motif; the list is empty")
    for index, motif in enumerate(motifs):
        if not isinstance(motif, str) or not motif:
            raise ValueError(f"motif {index + 1} is not a non-empty string: {motif!r}")
    motifs = [motif.upper() for motif in motifs]
    if len(set(motifs)) != len(motifs):
        repeated = sorted({m for m in motifs if motifs.count(m) > 1})
        raise ValueError(f"motifs repeat: {repeated}")

    outside = sorted({letter for motif in motifs for letter in motif} - set(alphabet))
    if outside:
        raise ValueError(
            f"motifs use {outside}, which are not in alphabet {alphabet!r}; "
            "pass a wider alphabet to count them")

    counts = [sum(1 for sequence in sequences if motif in sequence)
              for motif in motifs]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(motifs)))
    ax.barh(motifs, counts, color=colors, edgecolor='black')
    ax.set_xlabel("Count")
    ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.01, i, str(c), va='center', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig
