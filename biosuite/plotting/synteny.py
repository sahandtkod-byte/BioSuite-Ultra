"""
Synteny dotplot for whole-genome comparison.
Plots positional matches between two genomes as dots on a 2D grid.
Pure Python / matplotlib implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict


def compute_dotplot(seq1, seq2, word_size=10, step=1):
    """Compute a dotplot matrix by exact word matching.

    Args:
        seq1: first sequence (string).
        seq2: second sequence (string).
        word_size: k-mer size for matching.
        step: step size for sliding window.

    Returns:
        hits: list of (i, j) positions where k-mers match.
    """
    if len(seq1) < word_size or len(seq2) < word_size:
        return []

    # Build index for seq1
    index1 = defaultdict(list)
    for i in range(0, len(seq1) - word_size + 1, step):
        kmer = seq1[i:i + word_size].upper()
        index1[kmer].append(i)

    hits = []
    for j in range(0, len(seq2) - word_size + 1, step):
        kmer = seq2[j:j + word_size].upper()
        if kmer in index1:
            for i in index1[kmer]:
                hits.append((i, j))

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
        show_histograms: show marginal histograms.
    """
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

    if show_histograms and hits:
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
    """
    # Map gene -> position in each genome
    pos1 = {g: i for i, g in enumerate(genes1)}
    pos2 = {g: i for i, g in enumerate(genes2)}

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
    score = collinear / total_pairs if total_pairs > 0 else 0.0
    return score, pairs


def plot_synteny_dotplot(genes1, genes2, genome1_name="Genome 1",
                         genome2_name="Genome 2", title="Synteny Dotplot",
                         figsize=(8, 8), dot_color='#00ff88', ax=None):
    """Plot a synteny dotplot from gene orderings.

    Args:
        genes1: ordered gene names for genome 1.
        genes2: ordered gene names for genome 2.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Map gene -> position
    pos1 = {g: i for i, g in enumerate(genes1)}
    pos2 = {g: i for i, g in enumerate(genes2)}

    common = list(set(genes1) & set(genes2))
    if not common:
        ax.text(0.5, 0.5, "No orthologs found", ha='center', va='center',
                transform=ax.transAxes, color='gray')
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
    for g in common[:min(20, len(common))]:
        ax.annotate(g, (pos1[g], pos2[g]), fontsize=6, ha='center', va='bottom',
                    textcoords='offset points', xytext=(0, 5))

    ax.set_xlim(-0.5, len(genes1) - 0.5)
    ax.set_ylim(-0.5, len(genes2) - 0.5)
    ax.set_xlabel(genome1_name)
    ax.set_ylabel(genome2_name)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D
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
    """Plot a linear gene order diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if colors is None:
        cmap = plt.cm.tab20
        colors = [cmap(i / len(genes)) for i in range(len(genes))]

    n = len(genes)
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
    """Plot a full synteny map with gene order and connecting lines."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1],
                                    gridspec_kw={'hspace': 0.3})

    # Gene orders
    common = list(set(genes1) & set(genes2))
    pos1 = {g: i for i, g in enumerate(genes1)}
    pos2 = {g: i for i, g in enumerate(genes2)}

    # Color by shared genes
    cmap = plt.cm.Set3
    gene_colors = {}
    for i, g in enumerate(common):
        gene_colors[g] = cmap(i / max(len(common), 1))

    colors1 = [gene_colors.get(g, 'lightgray') for g in genes1]
    colors2 = [gene_colors.get(g, 'lightgray') for g in genes2]

    plot_gene_order(genes1, genome1_name, ax=ax1, colors=colors1)
    plot_gene_order(genes2, genome2_name, ax=ax2, colors=colors2)

    # Draw connecting lines
    from matplotlib.lines import Line2D
    for g in common:
        i1 = genes1.index(g)
        i2 = genes2.index(g)
        color = gene_colors.get(g, 'gray')
        ax1.annotate('', xy=(i2 + 0.5, 0), xycoords=ax2.get_yaxis_transform(),
                     xytext=(i1 + 0.5, 1), textcoords=ax1.get_yaxis_transform(),
                     arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=1.5))

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
