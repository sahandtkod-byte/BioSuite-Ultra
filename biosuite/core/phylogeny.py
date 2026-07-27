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
    a = np.array(list(seq1), dtype='U1')
    b = np.array(list(seq2), dtype='U1')
    valid = (a != '-') & (b != '-')
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0
    diff = ((a != b) & valid).sum()
    return float(diff / n_valid)

def distance_matrix(sequences):
    """Compute pairwise p-distance matrix for list of sequences (same length).

    Uses vectorized numpy comparisons for speed. For n sequences of
    length L, this avoids O(n^2 * L) Python-level iterations.
    """
    n = len(sequences)
    # Convert all sequences to a 2D numpy character array
    seq_arr = np.array([list(s) for s in sequences], dtype='U1')  # shape (n, L)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = seq_arr[i]
            b = seq_arr[j]
            valid = (a != '-') & (b != '-')
            n_valid = valid.sum()
            if n_valid > 0:
                d = float(((a != b) & valid).sum() / n_valid)
                mat[i][j] = mat[j][i] = d
    return mat

def upgma_tree(distance_mat, labels):
    """
    Build UPGMA tree using SciPy's linkage (method='average').
    Returns linkage matrix and dendrogram dictionary.
    """
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