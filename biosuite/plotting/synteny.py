"""
Synteny dotplot for whole-genome comparison.
Plots positional matches between two genomes as dots on a 2D grid.
Pure Python / matplotlib implementation.
"""
import operator
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'compute_dotplot',
    'plot_dotplot',
    'compute_synteny_score',
    'plot_synteny_dotplot',
    'plot_gene_order',
    'plot_synteny',
]

#: A dotplot of a repetitive sequence can produce millions of matches; past this
#: many the scatter is unreadable and the list itself is what exhausts memory.
_MAX_DOTPLOT_HITS = 1_000_000

#: Gene labels stop being readable well before a whole chromosome's worth.
_MAX_LABELLED_GENES = 20


def _checked_sequence(sequence, label):
    if not isinstance(sequence, str):
        raise ValueError(f"{label} must be a string, got {type(sequence).__name__}")
    return sequence.upper()


def _whole_number(value, label):
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{label} must be a whole number, got {value!r}") from None


def _gene_positions(genes, label):
    """Return the gene list (materialised) and a name -> position map."""
    genes = list(genes)
    positions = {}
    for index, gene in enumerate(genes):
        if not isinstance(gene, str) or not gene:
            raise ValueError(f"{label} entry {index + 1} is not a gene name: {gene!r}")
        if gene in positions:
            raise ValueError(
                f"{label} names {gene!r} twice; a gene order needs one position per gene")
        positions[gene] = index
    return genes, positions


def _checked_gene_order(genes, label):
    genes, positions = _gene_positions(genes, label)
    if not genes:
        raise ValueError(f"{label} is empty; a synteny plot needs at least one gene")
    return genes, positions


def compute_dotplot(seq1, seq2, word_size=10, step=1, max_hits=_MAX_DOTPLOT_HITS):
    """Compute a dotplot matrix by exact word matching.

    Args:
        seq1: first sequence (string).
        seq2: second sequence (string).
        word_size: k-mer size for matching.
        step: step size for sliding window.
        max_hits: largest number of matches to collect; a repetitive pair of
            sequences is refused rather than building an unreadable plot.

    Returns:
        hits: list of (i, j) positions where k-mers match.

    Raises:
        ValueError: if a word size, step or hit budget cannot be used.
    """
    seq1 = _checked_sequence(seq1, "seq1")
    seq2 = _checked_sequence(seq2, "seq2")
    word_size = _whole_number(word_size, "word_size")
    step = _whole_number(step, "step")
    max_hits = _whole_number(max_hits, "max_hits")
    if word_size < 1:
        raise ValueError(f"word_size must be at least 1, got {word_size}")
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    if max_hits < 1:
        raise ValueError(f"max_hits must be at least 1, got {max_hits}")

    if len(seq1) < word_size or len(seq2) < word_size:
        return []

    # Build index for seq1
    index1 = defaultdict(list)
    for i in range(0, len(seq1) - word_size + 1, step):
        kmer = seq1[i:i + word_size]
        index1[kmer].append(i)

    hits = []
    for j in range(0, len(seq2) - word_size + 1, step):
        kmer = seq2[j:j + word_size]
        for i in index1.get(kmer, ()):
            hits.append((i, j))
            if len(hits) > max_hits:
                raise ValueError(
                    f"word_size={word_size} finds more than {max_hits:,} matches "
                    f"between {len(seq1)}- and {len(seq2)}-base sequences; "
                    "raise word_size or step to thin the dotplot")

    return hits


def plot_dotplot(seq1, seq2, word_size=10, title="Dotplot",
                 seq1_name="Sequence 1", seq2_name="Sequence 2",
                 figsize=(8, 8), dot_size=2, dot_color='#00ff88',
                 ax=None, show_histograms=False):
    """Plot a sequence dotplot.

    Args:
        seq1, seq2: input sequences.
        word_size: k-mer size.
        title: plot title.
        seq1_name, seq2_name: axis labels.
        dot_size: marker size.
        dot_color: dot color.
        ax: optional matplotlib axes.
        show_histograms: show marginal histograms.  A supplied ``ax`` has no
            marginal axes, so the request is reported and skipped.

    Raises:
        ValueError: if either sequence is empty.
    """
    seq1 = _checked_sequence(seq1, "seq1")
    seq2 = _checked_sequence(seq2, "seq2")
    if not seq1 or not seq2:
        raise ValueError("plot_dotplot needs a non-empty sequence on each axis")

    hits = compute_dotplot(seq1, seq2, word_size)

    if ax is None:
        if show_histograms:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                   hspace=0.05, wspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax_histx = None
        ax_histy = None
        if show_histograms:
            print("Note: show_histograms needs marginal axes, so it is ignored "
                  "when an axes is supplied.")

    if hits:
        xs, ys = zip(*hits)
        ax.scatter(xs, ys, s=dot_size, c=dot_color, alpha=0.6, edgecolors='none')

    ax.set_xlim(0, len(seq1))
    ax.set_ylim(0, len(seq2))
    ax.set_xlabel(seq1_name)
    ax.set_ylabel(seq2_name)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_histograms and hits and ax_histx is not None:
        # X histogram
        ax_histx.hist(xs, bins=50, color=dot_color, alpha=0.7, edgecolor='black')
        ax_histx.set_ylabel("Count")
        ax_histx.tick_params(labelbottom=False)

        # Y histogram
        ax_histy.hist(ys, bins=50, orientation='horizontal', color=dot_color,
                      alpha=0.7, edgecolor='black')
        ax_histy.set_xlabel("Count")
        ax_histy.tick_params(labelleft=False)

    return fig


def compute_synteny_score(genes1, genes2):
    """Compute synteny score between two gene orderings.

    Args:
        genes1, genes2: ordered lists of gene names (orthologs).

    Returns:
        score: number of collinear gene pairs / total possible pairs.
        collinear_pairs: list of (gene, pos1, pos2) for collinear pairs.

    Raises:
        ValueError: if either order repeats a gene or holds a non-name.
    """
    genes1, pos1 = _gene_positions(genes1, "genes1")
    genes2, pos2 = _gene_positions(genes2, "genes2")

    common = set(genes1) & set(genes2)
    if len(common) < 2:
        return 0.0, []

    # Count collinear pairs (same order in both genomes)
    common_list = sorted(common, key=lambda g: pos1[g])
    collinear = 0
    pairs = []
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            g1, g2 = common_list[i], common_list[j]
            if pos2[g1] < pos2[g2]:
                collinear += 1
                pairs.append((g1, pos1[g1], pos2[g1]))

    total_pairs = len(common_list) * (len(common_list) - 1) / 2
    return collinear / total_pairs, pairs


def plot_synteny_dotplot(genes1, genes2, genome1_name="Genome 1",
                         genome2_name="Genome 2", title="Synteny Dotplot",
                         figsize=(8, 8), dot_color='#00ff88', ax=None):
    """Plot a synteny dotplot from gene orderings.

    Args:
        genes1: ordered gene names for genome 1.
        genes2: ordered gene names for genome 2.
        genome1_name, genome2_name: axis labels.
        title: plot title; when more than 20 genes are shared, the number of
            labelled genes is appended.
        figsize: figure size when no axes is supplied.
        dot_color: colour of a collinear gene.
        ax: optional matplotlib axes.

    Raises:
        ValueError: if either gene order is empty, repeats a name or holds a
            non-name.
    """
    genes1, pos1 = _checked_gene_order(genes1, "genes1")
    genes2, pos2 = _checked_gene_order(genes2, "genes2")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    common = sorted((g for g in pos1 if g in pos2), key=pos1.get)
    if not common:
        ax.text(0.5, 0.5, "No orthologs found", ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.set_xlabel(genome1_name)
        ax.set_ylabel(genome2_name)
        ax.set_title(title)
        return fig

    # Color collinear vs non-collinear
    xs, ys, colors = [], [], []
    for g in common:
        i, j = pos1[g], pos2[g]
        xs.append(i)
        ys.append(j)
        # Check if collinear with neighbors
        is_collinear = True
        for g2 in common:
            if g2 == g:
                continue
            i2, j2 = pos1[g2], pos2[g2]
            if (i < i2 and j > j2) or (i > i2 and j < j2):
                is_collinear = False
                break
        colors.append(dot_color if is_collinear else '#ff6b6b')

    ax.scatter(xs, ys, s=30, c=colors, edgecolors='black', linewidth=0.5, alpha=0.8)

    # Label some genes
    for g in common[:_MAX_LABELLED_GENES]:
        ax.annotate(g, (pos1[g], pos2[g]), fontsize=6, ha='center', va='bottom',
                    textcoords='offset points', xytext=(0, 5))

    if len(common) > _MAX_LABELLED_GENES:
        title = f"{title} (labels for {_MAX_LABELLED_GENES} of {len(common)} genes)"

    ax.set_xlim(-0.5, len(genes1) - 0.5)
    ax.set_ylim(-0.5, len(genes2) - 0.5)
    ax.set_xlabel(genome1_name)
    ax.set_ylabel(genome2_name)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dot_color,
               markersize=8, label='Collinear'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
               markersize=8, label='Non-collinear'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return fig


def plot_gene_order(genes, genome_name="Genome", figsize=(12, 2), ax=None,
                    colors=None):
    """Plot a linear gene order diagram.

    Raises:
        ValueError: if the gene list or a supplied colour list is empty.
    """
    genes = list(genes)
    if not genes:
        raise ValueError("plot_gene_order needs at least one gene")
    for index, gene in enumerate(genes):
        if not isinstance(gene, str) or not gene:
            raise ValueError(f"genes entry {index + 1} is not a gene name: {gene!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n = len(genes)
    if colors is None:
        cmap = plt.cm.tab20
        colors = [cmap(i / n) for i in range(n)]
    else:
        colors = list(colors)
        if not colors:
            raise ValueError(
                "plot_gene_order needs at least one colour when colours are supplied")

    for i, gene in enumerate(genes):
        rect = Rectangle((i, 0), 1, 1, facecolor=colors[i % len(colors)],
                          edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, gene, ha='center', va='center', fontsize=max(5, min(10, 100 // n)),
                fontweight='bold', color='white')

    ax.set_xlim(0, n)
    ax.set_ylim(-0.2, 1.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(genome_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig


def plot_synteny(genes1, genes2, genome1_name="Genome 1", genome2_name="Genome 2",
                 title="Synteny Map", figsize=(12, 6)):
    """Plot a full synteny map with gene order and connecting lines.

    Raises:
        ValueError: if either gene order is empty, repeats a name or holds a
            non-name.
    """
    genes1, pos1 = _checked_gene_order(genes1, "genes1")
    genes2, pos2 = _checked_gene_order(genes2, "genes2")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1],
                                    gridspec_kw={'hspace': 0.3})

    common = sorted((g for g in pos1 if g in pos2), key=pos1.get)

    # Color by shared genes
    cmap = plt.cm.Set3
    gene_colors = {g: cmap(i / len(common)) for i, g in enumerate(common)}

    colors1 = [gene_colors.get(g, 'lightgray') for g in genes1]
    colors2 = [gene_colors.get(g, 'lightgray') for g in genes2]

    plot_gene_order(genes1, genome1_name, ax=ax1, colors=colors1)
    plot_gene_order(genes2, genome2_name, ax=ax2, colors=colors2)

    # Draw connecting lines across the gap, from the bottom edge of the first
    # gene strip to the top edge of the second one.
    for g in common:
        i1, i2 = pos1[g], pos2[g]
        color = gene_colors[g]
        ax1.annotate('', xy=(i2 + 0.5, 1), xycoords=ax2.transData,
                     xytext=(i1 + 0.5, 0), textcoords=ax1.transData,
                     arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=1.5))

    fig.suptitle(title, fontsize=14)
    return fig
