"""
Distance matrix, UPGMA tree construction, and basic tree drawing.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from ..core.utils import apply_glass_ax

def p_distance(seq1, seq2):
    """Proportion of differing sites (ignoring gaps)."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be same length")
    diff = sum(1 for a,b in zip(seq1, seq2) if a != b and a != '-' and b != '-')
    total = sum(1 for a,b in zip(seq1, seq2) if a != '-' and b != '-')
    return diff/total if total>0 else 0

def distance_matrix(sequences):
    """Compute pairwise p-distance matrix for list of sequences (same length)."""
    n = len(sequences)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = p_distance(sequences[i], sequences[j])
            mat[i][j] = mat[j][i] = d
    return mat

def upgma_tree(distance_mat, labels):
    """
    Build UPGMA tree using SciPy's linkage (method='average').
    Returns linkage matrix and dendrogram dictionary.
    """
    from scipy.cluster.hierarchy import linkage
    linkage_mat = linkage(distance_mat, method='average', optimal_ordering=True)
    return linkage_mat

def plot_phylogenetic_tree(linkage_mat, labels, title="Phylogenetic Tree"):
    """Plot dendrogram from linkage matrix."""
    fig, ax = plt.subplots(figsize=(10,6))
    dendrogram(linkage_mat, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title)
    apply_glass_ax(ax)
    plt.tight_layout()
    return fig