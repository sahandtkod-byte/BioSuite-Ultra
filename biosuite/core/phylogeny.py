"""
Distance matrix, UPGMA tree construction, and basic tree drawing.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..core.utils import apply_glass_ax


def p_distance(seq1, seq2):
    """Proportion of differing sites (ignoring gaps).

    Comparison is case-insensitive: a soft-masked (lower-case) sequence is the
    same sequence, and a case-sensitive comparison measured 'ACGT' versus 'acgt'
    as 1.0 — maximally divergent.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be same length")
    a = np.array(list(seq1.upper()), dtype='U1')
    b = np.array(list(seq2.upper()), dtype='U1')
    valid = (a != '-') & (b != '-')
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0
    diff = ((a != b) & valid).sum()
    return float(diff / n_valid)

def distance_matrix(sequences):
    """Compute pairwise p-distance matrix for list of sequences (same length).

    Uses vectorized numpy comparisons for speed. For n sequences of
    length L, this avoids O(n^2 * L) Python-level iterations.  Like
    ``p_distance`` the comparison ignores case, so a soft-masked genome does not
    read as a maximally divergent taxon.

    Raises:
        ValueError: If the sequences differ in length — unaligned sequences have
            no defined per-site comparison.
    """
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        # numpy used to answer 'setting an array element with a sequence' from
        # inside its own array construction.
        raise ValueError(
            f"distance_matrix needs sequences of equal length, got lengths "
            f"{sorted(lengths)}; align them first (see biosuite.core.msa).")
    n = len(sequences)
    # Convert all sequences to a 2D numpy character array
    seq_arr = np.array([list(s.upper()) for s in sequences], dtype='U1')  # shape (n, L)
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

def upgma_tree(distance_mat):
    """
    Build a UPGMA linkage matrix with SciPy (method='average').

    Returns the linkage matrix only — the labels belong to the drawing step,
    ``plot_phylogenetic_tree(linkage_mat, labels)``, which is where they are used.

    The full square distance matrix is converted to scipy's condensed
    form explicitly — passing the square matrix directly emits a
    ClusterWarning and relies on implicit conversion behavior.
    """
    mat = np.asarray(distance_mat, dtype=float)
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        condensed = squareform(mat, checks=False)
    else:
        condensed = mat  # already condensed
    linkage_mat = linkage(condensed, method='average', optimal_ordering=True)
    return linkage_mat

def plot_phylogenetic_tree(linkage_mat, labels, title="Phylogenetic Tree"):
    """Plot dendrogram from linkage matrix."""
    fig, ax = plt.subplots(figsize=(10,6))
    dendrogram(linkage_mat, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title)
    apply_glass_ax(ax)
    plt.tight_layout()
    return fig