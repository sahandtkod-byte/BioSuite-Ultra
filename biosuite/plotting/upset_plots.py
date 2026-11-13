"""
UpSet plots for multi-set intersections.
Better than Venn diagrams for 3+ sets — shows all intersection sizes.
Pure Python implementation using matplotlib.
"""
import operator
from collections.abc import Mapping, Set as AbstractSet
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'compute_upset_matrix',
    'plot_upset',
    'upset_from_sets',
    'compute_set_statistics',
    'plot_set_sizes',
]

#: Every extra set doubles the table: 17 sets already mean 131,071 rows.
_MAX_SETS = 16


def _whole_number(value, label):
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{label} must be a whole number, got {value!r}") from None


def _checked_sets(sets_dict, label="sets_dict"):
    if not isinstance(sets_dict, Mapping):
        raise ValueError(
            f"{label} must be a mapping of set names to sets, "
            f"got {type(sets_dict).__name__}")
    for name, members in sets_dict.items():
        if not isinstance(members, AbstractSet):
            raise TypeError(
                f"set {name!r} must be a set of elements, got {type(members).__name__}")
    return sets_dict


def compute_upset_matrix(sets_dict):
    """Compute intersection sizes for all non-empty subsets.

    Args:
        sets_dict: dict mapping set_name -> set of elements.
                   Example: {'A': {1,2,3}, 'B': {2,3,4}, 'C': {3,4,5}}

    Returns:
        labels: list of sorted set names
        matrix: boolean matrix (2^n x n) where row i = binary repr of subset
        counts: exclusive intersection size for each row

    Raises:
        TypeError: if a value is not a set.
        ValueError: if the table would need more than 16 sets' worth of rows.
    """
    sets_dict = _checked_sets(sets_dict)
    labels = sorted(sets_dict.keys())
    n = len(labels)
    if n > _MAX_SETS:
        raise ValueError(
            f"compute_upset_matrix checks 2^n - 1 combinations; {n} sets need "
            f"{2 ** n - 1:,} rows, so it accepts at most {_MAX_SETS}. "
            "Pass fewer sets.")
    sets_list = [sets_dict[k] for k in labels]

    # Build all 2^n - 1 non-empty subsets with exclusive counts
    matrix = []
    counts = []
    for i in range(1, 2 ** n):
        bits = [(i >> bit) & 1 for bit in range(n)]
        # Intersection of all sets marked with 1
        included = [sets_list[j] for j in range(n) if bits[j] == 1]
        intersection = set.intersection(*included) if included else set()
        # Exclude elements that appear in any set marked with 0
        excluded = [sets_list[j] for j in range(n) if bits[j] == 0]
        if excluded:
            excluded_union = set.union(*excluded)
            intersection = intersection - excluded_union
        size = len(intersection)
        if size > 0:
            matrix.append(bits)
            counts.append(size)

    # Sort by count descending, then by mask
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    matrix = [matrix[i] for i in order]
    counts = [counts[i] for i in order]
    return labels, matrix, counts


def plot_upset(sets_dict, title="UpSet Plot", max_sets=10, figsize=(10, 7),
               bar_color='#00ff88', dot_color='#00cc66', ax=None):
    """Plot an UpSet diagram.

    The diagram is two panels (intersection sizes and the dot matrix) in a
    figure of its own.  ``ax`` is only usable when there is nothing to draw:
    then the "No intersections" note goes on it.  Passing ``ax`` together with
    data raises, because one axes cannot hold both panels.

    Args:
        sets_dict: dict mapping set name -> set of elements.
        title: plot title; when sets are dropped, the title records how many
            of how many were kept.
        max_sets: max number of sets to display (the largest are kept).
        figsize: figure size if ax is None.
        bar_color: color of the intersection size bars.
        dot_color: color of the dot matrix.
        ax: optional axes for the "No intersections" note.

    Returns:
        The figure, or None when every set is empty.

    Raises:
        ValueError: if the mapping is empty or the limit is not a positive
            whole number, or if a drawn diagram is asked to share one axes.
        TypeError: if a value is not a set.
    """
    sets_dict = _checked_sets(sets_dict)
    if not sets_dict:
        raise ValueError("plot_upset needs at least one set; the mapping is empty")
    max_sets = _whole_number(max_sets, "max_sets")
    if max_sets < 1:
        raise ValueError(f"max_sets must be at least 1, got {max_sets}")

    n_given = len(sets_dict)
    if n_given > max_sets:
        # Keep only the top sets by size
        set_sizes = {k: len(v) for k, v in sets_dict.items()}
        top = sorted(set_sizes, key=set_sizes.get, reverse=True)[:max_sets]
        sets_dict = {k: sets_dict[k] for k in top}

    labels, matrix, counts = compute_upset_matrix(sets_dict)

    n_sets = len(labels)
    n_rows = len(matrix)

    if n_rows == 0:
        if ax is not None:
            ax.text(0.5, 0.5, "No intersections", ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        return None

    if ax is not None:
        raise ValueError(
            "plot_upset draws a bar panel and a dot matrix; it cannot draw both "
            "on one supplied axes. Call it without ax.")

    if n_given > max_sets:
        title = f"{title} (top {max_sets} of {n_given} sets)"

    # Create layout: top = bars, bottom = dot matrix
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.6], hspace=0.05)

    # --- Top: bar chart of intersection sizes ---
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.bar(range(n_rows), counts, color=bar_color, edgecolor='black',
                      alpha=0.8, width=0.7)
    ax_bar.set_ylabel("Intersection size", fontsize=11)
    ax_bar.set_title(title, fontsize=13)
    ax_bar.set_xticks([])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    max_count = max(counts)
    for i, c in enumerate(counts):
        ax_bar.text(i, c + max_count * 0.02, str(c), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # --- Bottom: dot matrix ---
    ax_matrix = fig.add_subplot(gs[1])

    for row_idx in range(n_rows):
        for col_idx in range(n_sets):
            filled = matrix[row_idx][col_idx]
            color = dot_color if filled else 'lightgray'
            circle = plt.Circle((col_idx, row_idx), 0.35, fc=color,
                                ec='black' if filled else 'gray', lw=1)
            ax_matrix.add_patch(circle)
            # Draw vertical line if this set is part of the intersection
            if filled:
                ax_matrix.plot([col_idx, col_idx], [row_idx - 0.5, row_idx + 0.5],
                               color='gray', lw=1.5, alpha=0.5)

    ax_matrix.set_xlim(-0.5, n_sets - 0.5)
    ax_matrix.set_ylim(-0.5, n_rows - 0.5)
    ax_matrix.set_xticks(range(n_sets))
    ax_matrix.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax_matrix.set_yticks([])
    ax_matrix.spines['top'].set_visible(False)
    ax_matrix.spines['right'].set_visible(False)
    ax_matrix.spines['left'].set_visible(False)
    ax_matrix.set_ylabel("")

    return fig


def upset_from_sets(*sets, names=None):
    """Convenience: create UpSet from positional set arguments.

    Args:
        *sets: variable number of set-like objects.
        names: optional list of names for each set.

    Raises:
        ValueError: if no set is given, the names do not pair one-to-one with
            the sets, or a name repeats.
    """
    if not sets:
        raise ValueError("upset_from_sets needs at least one set")
    if names is None:
        names = [f"Set {i+1}" for i in range(len(sets))]
    else:
        names = list(names)
        if len(names) != len(sets):
            raise ValueError(
                f"upset_from_sets got {len(sets)} sets but {len(names)} name(s); "
                "pass one name per set")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"name {index + 1} is not a non-empty string: {name!r}")
    if len(set(names)) != len(names):
        repeated = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"set names repeat: {repeated}")
    d = {n: set(s) for n, s in zip(names, sets)}
    return plot_upset(d)


def compute_set_statistics(sets_dict):
    """Compute basic statistics for a collection of sets.

    Returns dict with: sizes, pairwise_jaccard, total_union, unique_per_set.

    Raises:
        TypeError: if a value is not a set.
    """
    sets_dict = _checked_sets(sets_dict)
    labels = sorted(sets_dict.keys())
    sizes = {k: len(v) for k, v in sets_dict.items()}
    total_union = len(set().union(*sets_dict.values()))
    total_intersection = set.intersection(*sets_dict.values()) if sets_dict else set()

    jaccard = {}
    for a, b in combinations(labels, 2):
        inter = len(sets_dict[a] & sets_dict[b])
        union = len(sets_dict[a] | sets_dict[b])
        jaccard[(a, b)] = inter / union if union > 0 else 0.0

    unique = {}
    for k in labels:
        others = set().union(*(sets_dict[j] for j in labels if j != k))
        unique[k] = len(sets_dict[k] - others)

    return {
        'sizes': sizes,
        'total_union': total_union,
        'total_intersection': len(total_intersection),
        'pairwise_jaccard': jaccard,
        'unique_per_set': unique,
    }


def plot_set_sizes(sets_dict, title="Set Sizes", figsize=(8, 4), ax=None):
    """Simple bar chart of set sizes.

    Returns the axes it drew on (the caller's axes when one is supplied).

    Raises:
        ValueError: if the mapping is empty.
    """
    if not sets_dict:
        raise ValueError("plot_set_sizes needs at least one set; the mapping is empty")
    labels = sorted(sets_dict.keys())
    sizes = [len(sets_dict[k]) for k in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    ax.bar(labels, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel("Size")
    ax.set_title(title)
    for i, s in enumerate(sizes):
        ax.text(i, s + max(sizes) * 0.02, str(s), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax
