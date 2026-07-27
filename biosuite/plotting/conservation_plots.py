"""
Conservation and sequence logo visualization.
Generates information-content-weighted sequence logos and conservation bars.
Pure Python / matplotlib implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter


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

    Returns:
        positions: list of position indices
        heights: dict of {char: np.array of heights per position}
        total_height: conservation height per position (bits)
    """
    max_len = max(len(s) for s in sequences)
    positions = list(range(max_len))
    n_seqs = len(sequences)

    all_heights = {c: np.zeros(max_len) for c in alphabet}
    total_height = np.zeros(max_len)

    for pos in range(max_len):
        counts = Counter()
        for seq in sequences:
            if pos < len(seq) and seq[pos] in alphabet:
                counts[seq[pos]] += 1

        freqs = _normalize_frequencies(counts)
        if not freqs:
            continue

        # Effective number of sequences at this position
        n_eff = sum(1 for s in sequences if pos < len(s) and s[pos] in alphabet)
        if n_eff == 0:
            continue

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
    """
    if color_scheme is None:
        color_scheme = {
            'A': '#22c55e',  # green
            'C': '#3b82f6',  # blue
            'G': '#eab308',  # yellow/orange
            'T': '#ef4444',  # red
            'U': '#ef4444',  # RNA uracil
        }

    positions, heights, total_h = compute_logo_heights(sequences, alphabet)

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
                    fontweight='bold', color=color,
                    transform=ax.get_xaxis_transform() if False else None)
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
    """
    positions, heights, total_h = compute_logo_heights(sequences, alphabet)
    max_bits = np.log2(len(alphabet))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    conservation = total_h / max_bits if max_bits > 0 else total_h

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
    """Combined logo + conservation bar plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.1})
    plot_sequence_logo(sequences, alphabet, title=title, ax=ax1)
    plot_conservation_bar(sequences, alphabet, title="", ax=ax2)
    return fig


def compute_conservation_scores(sequences, alphabet='ACGT'):
    """Return per-position conservation scores (0-1)."""
    positions, _, total_h = compute_logo_heights(sequences, alphabet)
    max_bits = np.log2(len(alphabet))
    return list(zip(positions, total_h / max_bits if max_bits > 0 else total_h))


def plot_motif_enrichment(sequences, motifs, alphabet='ACGT', title="Motif Enrichment",
                          figsize=(10, 4)):
    """Count occurrences of each motif and plot enrichment bar chart.

    Args:
        sequences: list of sequences.
        motifs: list of motif strings (e.g., ['ATG', 'TATAAA']).
    """
    counts = []
    for motif in motifs:
        c = sum(1 for s in sequences if motif in s)
        counts.append(c)

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
